import os
import io
import json
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from PIL import Image
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

# Load Google Drive API credentials
SERVICE_ACCOUNT_FILE = "herb-image-search-fd5324031218.json"  # Replace with your JSON file path
SCOPES = ["https://www.googleapis.com/auth/drive"]
credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
drive_service = build("drive", "v3", credentials=credentials)

# Google Drive folder containing models
GOOGLE_DRIVE_FOLDER_ID = "1pwZbzsHpnYK_seLqa2dn6VK6QKzI5LAP"  # Replace with your folder ID

# Temporary storage for downloaded models
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Function to list family folders in Google Drive
def get_families():
    query = f"'{GOOGLE_DRIVE_FOLDER_ID}' in parents and mimeType='application/vnd.google-apps.folder'"
    results = drive_service.files().list(q=query, fields="files(id, name)").execute()

    families_dict = {folder["name"]: folder["id"] for folder in results.get("files", [])}
    
    print("Retrieved Families from Drive:", families_dict)  # Debug print
    return {folder["name"]: folder["id"] for folder in results.get("files", [])}

# Function to get model file from Google Drive
def get_model_file(family_id, family_name):
    query = f"'{family_id}' in parents and name='{family_name}.h5'"
    results = drive_service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get("files", [])

    print(f"Searching for model '{family_name}.h5' in {family_id} -> Found: {files}")  # Debug print

    if files:
        return files[0]["id"]
    return None

# Function to download a file from Google Drive
def download_file(file_id, save_path):
    request = drive_service.files().get_media(fileId=file_id)
    with open(save_path, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()

# Function to load model dynamically
def load_model(family):
    model_path = os.path.join(MODEL_DIR, f"{family}.h5")

    # Download the model if not already downloaded
    if not os.path.exists(model_path):
        family_id = families[family]
        model_file_id = get_model_file(family_id, family)
        if model_file_id:
            download_file(model_file_id, model_path)

    # Load the model
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        model.make_predict_function()
        return model
    return None

# Function to get category labels from Google Drive
# def get_categories(family_id):
#     query = f"'{family_id}' in parents and mimeType='application/vnd.google-apps.folder' and name='1)dataset_for_model'"
#     results = drive_service.files().list(q=query, fields="files(id, name)").execute()
#     dataset_folder = results.get("files", [])
#     if not dataset_folder:
#         return []

#     dataset_id = dataset_folder[0]["id"]
#     query = f"'{dataset_id}' in parents and mimeType='application/vnd.google-apps.folder'"
#     results = drive_service.files().list(q=query, fields="files(name)").execute()
#     return sorted([folder["name"] for folder in results.get("files", [])])


def get_categories(family_id):
    # Get dataset folder (1)dataset_for_model)
    query = f"'{family_id}' in parents and mimeType='application/vnd.google-apps.folder' and name='1)dataset_for_model'"
    results = drive_service.files().list(q=query, fields="files(id, name)").execute()
    dataset_folder = results.get("files", [])
    if not dataset_folder:
        return []

    dataset_id = dataset_folder[0]["id"]

    # Get 'train' folder inside dataset
    query = f"'{dataset_id}' in parents and mimeType='application/vnd.google-apps.folder' and name='train'"
    results = drive_service.files().list(q=query, fields="files(id, name)").execute()
    train_folder = results.get("files", [])
    if not train_folder:
        return []

    train_folder_id = train_folder[0]["id"]

    # Get species names (folders inside 'train')
    query = f"'{train_folder_id}' in parents and mimeType='application/vnd.google-apps.folder'"
    results = drive_service.files().list(q=query, fields="files(name)").execute()
    
    return sorted([folder["name"] for folder in results.get("files", [])])




# Function to classify image
def classify_image(imageFile, model, categories):
    img = Image.open(imageFile)
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    pred = model.predict(x)
    categoryValue = np.argmax(pred, axis=1)[0]

    print("Categories list:", categories)
    print("Predicted Index:", categoryValue)

    return categories[categoryValue] if categories else "Unknown"

# Get all families from Google Drive
families = get_families()

# Routes
@app.route("/")
def main():
    return render_template("index2.html", families=families.keys())

@app.route("/family/<family>")
def family_classification(family):
    if family not in families:
        return "Invalid Family", 404
    return render_template("classify.html", family=family)

@app.route("/submit/<family>", methods=["POST"])
def get_output(family):
    if family not in families:
        return "Invalid Family", 404

    if "my_image" not in request.files:
        return "No file uploaded", 400

    img = request.files["my_image"]
    if img.filename == "":
        return "No selected file", 400

    # Save uploaded image temporarily
    UPLOAD_FOLDER = "uploaded_images"
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    img_path = os.path.join(UPLOAD_FOLDER, img.filename).replace("\\", "/")
    img.save(img_path)

    # Load model and categories
    model = load_model(family)
    if not model:
        return "Model not found", 500

    categories = get_categories(families[family])

    # Perform classification
    prediction = classify_image(img_path, model, categories)

    return render_template("classify.html", prediction=prediction, img_path=img_path, family=family)

if __name__ == "__main__":
    app.run(debug=True)
