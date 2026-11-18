import json
import joblib
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import torch
import torchxrayvision as xrv
import torchvision


# =========================================================
# PATHS
# =========================================================

CONV_WEIGHTS = "models/convnext_pneumonia.weights.h5"
EFF_WEIGHTS = "models/efficientnet_pneumonia.weights.h5"
CALIBRATOR_PATH = "models/pneumonia_calibrators.pkl"
CONFIG_PATH = "models/pneumonia_ensemble_config.json"


# =========================================================
# FOCAL LOSS
# =========================================================

def focal_loss(gamma=2.0, alpha=0.75):

    def loss_function(y_true, y_pred):

        y_true = tf.cast(y_true, tf.float32)

        y_pred = tf.clip_by_value(
            y_pred,
            1e-7,
            1.0 - 1e-7
        )

        cross_entropy = -(
            y_true * tf.math.log(y_pred)
            +
            (1.0 - y_true)
            * tf.math.log(1.0 - y_pred)
        )

        p_t = (
            y_true * y_pred
            +
            (1.0 - y_true)
            * (1.0 - y_pred)
        )

        alpha_factor = (
            y_true * alpha
            +
            (1.0 - y_true) * (1.0 - alpha)
        )

        modulating_factor = tf.pow(
            1.0 - p_t,
            gamma
        )

        loss = (
            alpha_factor
            * modulating_factor
            * cross_entropy
        )

        return tf.reduce_mean(loss)

    return loss_function


# =========================================================
# CONVNEXT
# =========================================================

def build_convnext():

    base_model = keras.applications.ConvNeXtTiny(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        pooling="avg",
        include_preprocessing=True
    )

    base_model.trainable = False

    inputs = keras.Input(
        shape=(224, 224, 3)
    )

    x = layers.RandomRotation(0.03)(inputs)
    x = layers.RandomZoom(0.08)(x)
    x = layers.RandomContrast(0.10)(x)

    x = base_model(
        x,
        training=False
    )

    x = layers.Dropout(0.40)(x)

    x = layers.Dense(
        128,
        activation="relu"
    )(x)

    x = layers.Dropout(0.25)(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid"
    )(x)

    model = keras.Model(
        inputs,
        outputs,
        name="ConvNeXt_Pneumonia"
    )

    return model


# =========================================================
# EFFICIENTNET
# =========================================================

def build_efficientnet():

    base_model = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        pooling="avg"
    )

    base_model.trainable = False

    inputs = keras.Input(
        shape=(224, 224, 3)
    )

    x = layers.RandomRotation(0.03)(inputs)
    x = layers.RandomZoom(0.08)(x)
    x = layers.RandomContrast(0.15)(x)

    x = base_model(
        x,
        training=False
    )

    x = layers.Dropout(0.40)(x)

    x = layers.Dense(
        128,
        activation="relu"
    )(x)

    x = layers.Dropout(0.25)(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid"
    )(x)

    model = keras.Model(
        inputs,
        outputs,
        name="EfficientNet_Pneumonia"
    )

    return model


# =========================================================
# LOAD MODELS
# =========================================================

print("Loading ConvNeXt...")

final_conv = build_convnext()
final_conv.load_weights(CONV_WEIGHTS)

print("ConvNeXt loaded.")


print("Loading EfficientNet...")

final_eff = build_efficientnet()
final_eff.load_weights(EFF_WEIGHTS)

print("EfficientNet loaded.")


# =========================================================
# LOAD XRV
# =========================================================

print("Loading TorchXRayVision...")

device = torch.device("cpu")

xrv_model = xrv.models.DenseNet(
    weights="densenet121-res224-rsna"
)

xrv_model = xrv_model.to(device)
xrv_model.eval()

pneumonia_index = xrv_model.pathologies.index(
    "Pneumonia"
)

xrv_transform = torchvision.transforms.Compose([
    xrv.datasets.XRayCenterCrop(),
    xrv.datasets.XRayResizer(224)
])

print("XRV loaded.")


# =========================================================
# LOAD CONFIGURATION
# =========================================================

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

calibrators = joblib.load(
    CALIBRATOR_PATH
)

W_CONV = config["weights"]["convnext"]
W_EFF = config["weights"]["efficientnet"]
W_XRV = config["weights"]["xrv"]

THRESHOLD = config["threshold"]

print("Model configuration loaded.")
print("Threshold:", THRESHOLD)


# =========================================================
# XRV PREDICTION
# =========================================================

def xrv_predict_one(image_path):

    image = xrv.utils.load_image(
        image_path
    )

    image = xrv_transform(image)

    image_tensor = torch.from_numpy(
        image
    ).unsqueeze(0).float()

    image_tensor = image_tensor.to(device)

    with torch.no_grad():

        output = xrv_model(
            image_tensor
        )

    probability = float(
        output[0][pneumonia_index]
        .cpu()
        .numpy()
    )

    return probability


# =========================================================
# FINAL PREDICTION
# =========================================================

def predict_uploaded_xray(image_path):

    # -------------------------------
    # TensorFlow preprocessing
    # -------------------------------

    image = tf.io.read_file(
        image_path
    )

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(
        image,
        (224, 224)
    )

    image = tf.cast(
        image,
        tf.float32
    )

    image_batch = tf.expand_dims(
        image,
        axis=0
    )


    # -------------------------------
    # ConvNeXt
    # -------------------------------

    conv_raw = float(
        final_conv.predict(
            image_batch,
            verbose=0
        )[0][0]
    )


    # -------------------------------
    # EfficientNet
    # -------------------------------

    eff_raw = float(
        final_eff.predict(
            image_batch,
            verbose=0
        )[0][0]
    )


    # -------------------------------
    # XRV
    # -------------------------------

    xrv_raw = xrv_predict_one(
        image_path
    )


    # -------------------------------
    # Calibration
    # -------------------------------

    conv_cal = float(
        calibrators["conv"]
        .predict_proba(
            [[conv_raw]]
        )[0][1]
    )

    eff_cal = float(
        calibrators["eff"]
        .predict_proba(
            [[eff_raw]]
        )[0][1]
    )

    xrv_cal = float(
        calibrators["xrv"]
        .predict_proba(
            [[xrv_raw]]
        )[0][1]
    )


    # -------------------------------
    # Ensemble
    # -------------------------------

    final_probability = (
        W_CONV * conv_cal
        +
        W_EFF * eff_cal
        +
        W_XRV * xrv_cal
    )


    # -------------------------------
    # Classification
    # -------------------------------

    if final_probability >= THRESHOLD:

        prediction = "PNEUMONIA"

    else:

        prediction = "NORMAL"


    confidence = (
        final_probability
        if prediction == "PNEUMONIA"
        else 1.0 - final_probability
    )


    # -------------------------------
    # Return
    # -------------------------------

    return {
        "prediction": prediction,
        "confidence": confidence,
        "pneumonia_probability": final_probability,
        "normal_probability": 1.0 - final_probability,
        "convnext_probability": conv_cal,
        "efficientnet_probability": eff_cal,
        "xrv_probability": xrv_cal
    }