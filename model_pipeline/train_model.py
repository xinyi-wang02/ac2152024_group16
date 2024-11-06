import argparse
import os
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.model_selection import train_test_split
from PIL import Image


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
    def __init__(self, csv_file, images_folder, transform=None):
        self.df = pd.read_csv(csv_file)
        self.images_folder = images_folder
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.images_folder, self.df.iloc[idx]['image_name'])
        image = Image.open(img_name).convert("RGB")
        label = int(self.df.iloc[idx]['label_encoded'])

        if self.transform:
            image = self.transform(image)

        return { 'image' : image, 'label':label, 'img_name': img_name }
    
def create_dataloader(loader_csv_path, all_images_folder, batch_size):
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

    dataset = ImageDataset(csv_file=loader_csv_path, images_folder=all_images_folder, transform=transform)
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



def main():
    parser = argparse.ArgumentParser(description="Train a car model recognition system.")
    parser.add_argument('--csv_path', type=str, default="/app/subset_car_images_mock.csv", help='Path to the original CSV.')
    parser.add_argument('--train_csv_path', type=str, default="/app/train_class_label.csv", help='Path to save the train CSV.')
    parser.add_argument('--test_csv_path', type=str, default="/app/test_class_label.csv", help='Path to save the test CSV.')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of the data for testing.')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--all_images_folder', type=str, default="/app/all_images_mock", help='Folder containing all images.')
    parser.add_argument('--train_iteration_folder', type=str, default="/app/train_iteration", help='Folder to save the iteration CSV.')
    parser.add_argument('--num_img', type=int, default=10, help='Number of images to sample.')
    parser.add_argument('--train_or_not', default=True, help='Flag to determine train or test subset.')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size for DataLoader.')
    parser.add_argument('--num_classes', type=int, default=196, help='Number of output classes.')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the optimizer.')

    args = parser.parse_args()

    args = parser.parse_args()

    split_csv(args.csv_path, args.train_csv_path, args.test_csv_path, args.test_size, args.random_state)
    loader_csv_path = create_subset_csv(
        args.train_csv_path if args.train_or_not else args.test_csv_path,
        args.all_images_folder,
        args.train_iteration_folder,
        args.num_img,
        args.train_or_not
    )
    train_loader = create_dataloader(args.train_csv_path, args.all_images_folder, args.batch_size)
    fine_tune_resnet(train_loader, args.num_classes, args.num_epochs, args.learning_rate)

if __name__ == "__main__":
    main()