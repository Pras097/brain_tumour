from flask import Flask, render_template, request
import os
import cv2
import joblib
import numpy as np

from skimage.feature import (
    hog,
    local_binary_pattern,
    graycomatrix,
    graycoprops
)
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
print("Loading Model...")

model = joblib.load("Best_BrainTumor_Model.pkl")
scaler = joblib.load("Scaler.pkl")
encoder = joblib.load("LabelEncoder.pkl")

print("Model Loaded Successfully!")
IMG_SIZE = 128

RADIUS = 2
N_POINTS = 8 * RADIUS
def preprocess_image(image_path):

    image = cv2.imread(image_path)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return gray
def extract_hog(image):

    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        feature_vector=True
    )

    return features
def extract_lbp(image):

    lbp = local_binary_pattern(
        image,
        N_POINTS,
        RADIUS,
        method='uniform'
    )

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, N_POINTS + 3),
        range=(0, N_POINTS + 2)
    )

    hist = hist.astype("float")

    hist /= (hist.sum() + 1e-7)

    return hist
def extract_glcm(image):

    glcm = graycomatrix(
        image,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    asm = graycoprops(glcm, 'ASM')[0, 0]

    return np.array([
        contrast,
        dissimilarity,
        homogeneity,
        energy,
        correlation,
        asm
    ])
def extract_features(image_path):

    image = preprocess_image(image_path)

    hog_features = extract_hog(image)

    lbp_features = extract_lbp(image)

    glcm_features = extract_glcm(image)

    features = np.concatenate([
        hog_features,
        lbp_features,
        glcm_features
    ])

    return features
def predict_brain_tumor(image_path):

    features = extract_features(image_path)

    features = features.reshape(1, -1)

    features = scaler.transform(features)

    prediction = model.predict(features)

    probability = model.predict_proba(features)

    confidence = float(np.max(probability) * 100)

    predicted_class = encoder.inverse_transform(prediction)[0]

    return predicted_class, confidence
@app.route("/")
def home():

    return render_template(
        "index.html",
        prediction=None
    )
@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:

        return render_template(
            "index.html",
            error="Please upload an image."
        )

    file = request.files["image"]

    if file.filename == "":

        return render_template(
            "index.html",
            error="Please select an image."
        )

    filename = file.filename

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(filepath)

    prediction, confidence = predict_brain_tumor(filepath)

    if prediction.lower() == "notumor":
        badge = "success"

    elif prediction.lower() == "glioma":
        badge = "danger"

    elif prediction.lower() == "meningioma":
        badge = "warning"

    else:
        badge = "primary"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=round(confidence, 2),
        image=filepath,
        badge=badge
    )
if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )