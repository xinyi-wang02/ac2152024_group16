import os
import io
import torch
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from torchvision import models, transforms
import pandas as pd

# Suppress TensorFlow logging (if using in the same environment)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the model
def load_model(model_path='model.pth', num_classes=196):
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Load label names from names.csv
def load_label_names(names_path='names.csv'):
    names_df = pd.read_csv(names_path, header=None)
    label_names = {idx: name for idx, name in enumerate(names_df[0], start=1)}
    return label_names

# Preprocessing transformations
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def transform_image(pillow_image):
    """
    Applies preprocessing transformations to a Pillow image.
    """
    image = image_transforms(pillow_image).unsqueeze(0)  # Add batch dimension
    return image

def predict(model, image_tensor, label_names):
    """
    Predicts the label for a preprocessed image tensor.
    """
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_idx = torch.max(outputs, 1)
        predicted_label = label_names[predicted_idx.item() + 1]  # Adjust index as necessary
    return predicted_label

# Initialize Flask app and load model and labels
app = Flask(__name__)
model = load_model('/Users/smallina/Desktop/Stanford/model.pth')
label_names = load_label_names('/Users/smallina/Desktop/Stanford/Data/names.csv')

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({"error": "no file"})

        try:
            image_bytes = file.read()
            pillow_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            tensor = transform_image(pillow_img)
            prediction = predict(model, tensor, label_names)
            data = {"prediction": prediction}
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)})

    return "OK"

if __name__ == "__main__":
    app.run(debug=True)