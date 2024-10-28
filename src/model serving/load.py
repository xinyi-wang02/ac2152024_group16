import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms, models

# Load the pretrained resnet18 model and adjust for the number of classes
def load_model(model_path='model.pth', num_classes=196):
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)  # Update final layer to match number of classes
    model.load_state_dict(torch.load(model_path))
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

def preprocess_image(img_path):
    image = Image.open(img_path).convert("RGB")
    image = image_transforms(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def predict_label(model, img_path, label_names):
    image = preprocess_image(img_path)

    with torch.no_grad():
        outputs = model(image)
        _, predicted_idx = torch.max(outputs, 1)
        predicted_label = label_names[predicted_idx.item() + 1]  # Adjust index as necessary

    return predicted_label

if __name__ == "__main__":
    model_path = '/Users/smallina/Desktop/Stanford/model.pth'
    names_path = '/Users/smallina/Desktop/Stanford/Data/names.csv'
    input_image_path = '/Users/smallina/Desktop/Stanford/Data/cars_test/00001.jpg'

    model = load_model(model_path)
    label_names = load_label_names(names_path)

    label = predict_label(model, input_image_path, label_names)
    print(f"Predicted label: {label}")