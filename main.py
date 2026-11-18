from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import os
import uuid
import shutil

from model_service import predict_uploaded_xray



# .\.venv\Scripts\Activate.ps1
# python -m uvicorn main:app --host 0.0.0.0 --port 8000

app = FastAPI(
    title="Pneumonia Detection API"
)
@app.on_event("startup")
async def startup_message():
    print("\n" + "=" * 60)
    print("     PNEUMONIA DETECTION API STARTED")
    print("=" * 60)
    print("API:  http://127.0.0.1:8000")
    print("DOCS: http://127.0.0.1:8000/docs")
    print("=" * 60 + "\n")


# CORS


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# TEMP DIRECTORY


TEMP_DIR = "temp"

os.makedirs(
    TEMP_DIR,
    exist_ok=True
)


# HEALTH CHECK


@app.get("/")
def home():

    return {
        "message": "Pneumonia Detection API is running"
    }


# PREDICT


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # Check image type

    if not file.content_type:
        raise HTTPException(
            status_code=400,
            detail="File type missing"
        )

    allowed_types = [
        "image/jpeg",
        "image/png",
        "image/jpg"
    ]

    if file.content_type not in allowed_types:

        raise HTTPException(
            status_code=400,
            detail="Only JPG and PNG images are allowed"
        )


    # Generate temporary filename

    extension = ".jpg"

    if file.content_type == "image/png":
        extension = ".png"

    filename = (
        str(uuid.uuid4())
        + extension
    )

    file_path = os.path.join(
        TEMP_DIR,
        filename
    )


    try:

        # Save uploaded image

        with open(
            file_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )


        # Run ML model

        result = predict_uploaded_xray(
            file_path
        )


        return {
            "success": True,
            "result": result
        }


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


    finally:

        # Delete temporary image

        if os.path.exists(file_path):

            os.remove(file_path)


# Deploy on Render
# In Render, choose New → Web Service and connect the GitHub repository.
# Select Python 3. Use:
# Build Command: pip install -r requirements.txt
# Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT
# Deploy, then test the service URL and its /docs page.