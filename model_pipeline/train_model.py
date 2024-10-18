import gc
import os
import io
import pandas as pd
import argparse
from PIL import Image
from matplotlib import pyplot as plt
from numpy import unravel_index

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from google.cloud import storage


def split_csv(csv_path, train_csv_path, test_csv_path, test_size=0.2, random_state=42):
    """
    Splits the original CSV into train and test CSVs based on the specified test size.

    Args:
        csv_path (str): Path to the original CSV file.
        train_csv_path (str): Path to save the train CSV.
        test_csv_path (str): Path to save the test CSV.
        test_size (float): Proportion of the data to include in the test split (default=0.2).
        random_state (int): Seed for reproducibility (default=42).

    Returns:
        None
    """
    df = pd.read_csv(csv_path)

    # Perform train-test split
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    # Save the train and test CSVs
    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)

    print(f"Train CSV saved at: {train_csv_path}, Size: {len(train_df)}")
    print(f"Test CSV saved at: {test_csv_path}, Size: {len(test_df)}")

def create_subset_csv(subset_csv_path, all_images_folder, train_iteration_folder, num_img, train_or_not=True):
    """
    Creates a subset of the original CSV with 'num_img' random images and their labels,
    and saves it in the specified train iteration folder.

    Args:
        subset_csv_path (str): Path to the train/test CSV file.
        all_images_folder (str): Path to the folder containing all images.
        train_iteration_folder (str): Path to save the subset CSV.
        num_img (int): Number of images to randomly select.
        train_or_not (bool): If True, save as 'train_subset.csv'; otherwise, save as 'test_subset.csv'.

    Returns:
        str: Path to the saved subset CSV file.
    """
    df = pd.read_csv(subset_csv_path)

    valid_images = [
        img for img in df['image_name'] if os.path.isfile(os.path.join(all_images_folder, img))
    ]
    df_filtered = df[df['image_name'].isin(valid_images)]

    # Randomly select images
    if num_img > len(df_filtered):
        raise ValueError(f"Requested {num_img} images, but only {len(df_filtered)} are available.")
    
    df_subset = df_filtered.sample(n=num_img, random_state=42).reset_index(drop=True)
    os.makedirs(train_iteration_folder, exist_ok=True)

    # Determine the filename based on whether it's train or test data
    subset_type = 'train_subset.csv' if train_or_not else 'test_subset.csv'
    subset_path = os.path.join(train_iteration_folder, subset_type)

    df_subset.to_csv(subset_path, index=False)
    print(f"Subset CSV saved at {subset_path}")

    return subset_path

class ImageDataset(Dataset):
    """
    Custom Dataset for loading images and labels from the subset CSV.
    """
    def __init__(self, csv_file, bucket_name, transform=None):
        self.df = pd.read_csv(csv_file)
        self.bucket_name = bucket_name
        self.client = storage.Client()
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def read_image_from_gcs(self, image_path):
        """Reads an image from GCS."""
        bucket = self.client.bucket(self.bucket_name)
        blob = bucket.blob(image_path)
        image_bytes = blob.download_as_bytes()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return image
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.images_folder, self.df.iloc[idx]['image_name'])
        image = self.read_image_from_gcs(img_name)
        label = int(self.df.iloc[idx]['label_encoded'])

        if self.transform:
            image = self.transform(image)

        return { 'image' : image, 'label':label, 'img_name': img_name }
    
def create_dataloader(loader_csv_path, bucket_name, all_images_folder, batch_size):
    """
    Creates a DataLoader from the subset CSV file.

    Args:
        csv_path (str): Path to the subset CSV file.
        all_images_folder (str): Path to the folder containing all images.
        batch_size (int): Batch size for DataLoader.

    Returns:
        DataLoader: PyTorch DataLoader object.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the images
        transforms.ToTensor(),          # Convert images to PyTorch tensors
    ])

    dataset = ImageDataset(csv_file=loader_csv_path, bucket_name=bucket_name, images_folder=all_images_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

def fine_tune_resnet(train_loader, num_classes=195, num_epochs=5, learning_rate=0.0001):
    """
    Fine-tune a pretrained ResNet model for car model classification.

    Args:
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader, optional): DataLoader for validation data.
        num_classes (int): Number of output classes (car models).
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for optimizer.

    Returns:
        model: The fine-tuned ResNet model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True)
    
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()  # Set model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        for batch in tqdm(train_loader, desc="Training", leave=False):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()  
            outputs = model(images) 
            loss = criterion(outputs, labels) 

            loss.backward()  
            optimizer.step()  

            # Track statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        print(f"Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    print("Training Complete.")
    return model



csv_path = "/cs215_car_dataset_w_class/car_preprocessed_folder/class_label.csv"
train_csv_path = "/cs215_car_dataset_w_class/car_preprocessed_folder/train_class_label.csv"
test_csv_path = "/cs215_car_dataset_w_class/car_preprocessed_folder/test_class_label.csv"
test_size = 0.2
random_state = 42

subset_csv_path = train_csv_path
all_images_folder = "/cs215_car_dataset_w_class/car_preprocessed_folder/all_images"
train_iteration_folder = "/cs215_car_dataset_w_class/train_iteration"
num_img = 128
train_or_not = True

loader_csv_path = subset_csv_path
bucket_name = "cs215_car_dataset_w_class"
all_images_folder = "/cs215_car_dataset_w_class/car_preprocessed_folder/all_images"
batch_size = 32

num_classes=195
num_epochs=5
learning_rate=0.0001

split_csv(csv_path, train_csv_path, test_csv_path, test_size, random_state)
create_subset_csv(subset_csv_path, all_images_folder, train_iteration_folder, num_img, train_or_not)
train_loader = create_dataloader(loader_csv_path, bucket_name, all_images_folder, batch_size)
fine_tune_resnet(train_loader, num_classes, num_epochs, learning_rate)
