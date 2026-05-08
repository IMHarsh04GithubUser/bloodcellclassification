import os
import torch
import torch.nn as nn
import numpy as np
import joblib

from flask import Flask, request, jsonify
from flask_cors import CORS

from torchvision import models, transforms
from torchvision.models import efficientnet_b0, mobilenet_v2

from PIL import Image

# ---------------- FLASK APP ---------------- #
app = Flask(__name__)
CORS(app)

# ---------------- DEVICE ---------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- BASE PATH ---------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- LOAD CLASS NAMES ---------------- #
class_names = joblib.load(
    os.path.join(BASE_DIR, "class_names.pkl")
)

num_classes = len(class_names)

# ---------------- LOAD RESNET ---------------- #
resnet_model = models.resnet50(weights=None)

resnet_model.fc = nn.Sequential(
    nn.Linear(resnet_model.fc.in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, num_classes)
)

resnet_model.load_state_dict(
    torch.load(
        os.path.join(BASE_DIR, "resnet_model.pth"),
        map_location=device
    )
)

resnet_model.to(device)
resnet_model.eval()

# ---------------- LOAD EFFICIENTNET ---------------- #
eff_model = efficientnet_b0(weights=None)

eff_model.classifier[1] = nn.Linear(
    eff_model.classifier[1].in_features,
    num_classes
)

eff_model.load_state_dict(
    torch.load(
        os.path.join(BASE_DIR, "efficientnet_model.pth"),
        map_location=device
    )
)

eff_model.to(device)
eff_model.eval()

# ---------------- LOAD MOBILENET ---------------- #
mob_model = mobilenet_v2(weights=None)

mob_model.classifier[1] = nn.Linear(
    mob_model.last_channel,
    num_classes
)

mob_model.load_state_dict(
    torch.load(
        os.path.join(BASE_DIR, "mobilenet_model.pth"),
        map_location=device
    )
)

mob_model.to(device)
mob_model.eval()

# ---------------- LOAD META MODEL ---------------- #
meta_model = joblib.load(
    os.path.join(BASE_DIR, "meta_model.pkl")
)

# ---------------- IMAGE TRANSFORM ---------------- #
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ---------------- PREDICTION FUNCTION ---------------- #
def predict_image(image):

    image = image.convert("RGB")

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        r = torch.softmax(
            resnet_model(image),
            dim=1
        )

        e = torch.softmax(
            eff_model(image),
            dim=1
        )

        m = torch.softmax(
            mob_model(image),
            dim=1
        )

    # Combine features
    features = torch.cat(
        (r, e, m),
        dim=1
    ).cpu().numpy()

    # Meta prediction
    pred = meta_model.predict(features)[0]

    return class_names[pred]

# ---------------- API ROUTE ---------------- #
@app.route("/predict", methods=["POST"])
def predict():

    try:

        if "image" not in request.files:
            return jsonify({
                "error": "No image uploaded"
            }), 400

        file = request.files["image"]

        image = Image.open(file.stream)

        prediction = predict_image(image)

        return jsonify({
            "prediction": prediction
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

# ---------------- HOME ROUTE ---------------- #
@app.route("/")
def home():

    return "Blood Cell ML API Running Successfully"

# ---------------- RUN ---------------- #
if __name__ == "__main__":
    app.run(debug=True)
