from flask import Flask, request, render_template
import numpy as np
import pickle
import pandas as pd
import os
import requests

# Google Drive file IDs (Replace these with your actual file IDs)
MODEL_FILE_ID = "https://drive.google.com/file/d/13cF0dmkd1Jx15q6mDXfVzXwIi7uPEclD/view?usp=sharing"
PREPROCESSOR_FILE_ID = "https://drive.google.com/file/d/1hT4LANqLgObXz1hLaWeOguziPGDPxGFv/view?usp=sharing"

# Function to download files from Google Drive
def download_file_from_google_drive(file_id, dest_path):
    URL = "https://drive.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={"id": file_id}, stream=True)
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(1024):
            if chunk:
                f.write(chunk)

# Download models if they don't exist
if not os.path.exists("dtr.pkl"):
    download_file_from_google_drive(MODEL_FILE_ID, "dtr.pkl")

if not os.path.exists("preprocessor.pkl"):
    download_file_from_google_drive(PREPROCESSOR_FILE_ID, "preprocessor.pkl")

# Load models
dtr = pickle.load(open("dtr.pkl", "rb"))
preprocessor = pickle.load(open("preprocessor.pkl", "rb"))
df = pd.read_csv("yield_df.csv")

# Initialize Flask app
app = Flask(__name__)

# Home page
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    areas = sorted(df["Area"].dropna().unique().tolist())  # Ensure correct column name
    items = sorted(df["Item"].dropna().unique().tolist())  # Ensure correct column name

    if request.method == "POST":
        Year = request.form["Year"]
        average_rain_fall_mm_per_year = request.form["average_rain_fall_mm_per_year"]
        pesticides_tonnes = request.form["pesticides_tonnes"]
        avg_temp = request.form["avg_temp"]
        Area = request.form["Area"]
        Item = request.form["Item"]

        features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)
        transformed_features = preprocessor.transform(features)
        prediction = dtr.predict(transformed_features).reshape(1, -1)

        return render_template("predict.html", prediction=prediction[0][0], areas=areas, items=items)

    return render_template("predict.html", prediction=None, areas=areas, items=items)

# Contact Page
@app.route("/contact")
def contact():
    return render_template("contact.html")

if __name__ == "__main__":
    app.run(debug=True)
