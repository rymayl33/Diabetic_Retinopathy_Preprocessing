from sklearn.metrics import precision_score, recall_score, f1_score
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import cv2
import numpy as np
from skimage.filters import frangi, hessian
from PIL import Image
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))

class DiabeticRetinopathyDataset(Dataset):
    def __init__(self, dataframe, png_dir, jpeg_dir, transform=None, preprocess=None):
        self.data = dataframe.reset_index(drop=True)
        self.png_dir = png_dir
        self.jpeg_dir = jpeg_dir
        self.transform = transform
        self.preprocess = preprocess

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_filename = self.data.iloc[idx]['image'].strip()
        label = self.data.iloc[idx]['level']
        
        img_path = self.jpeg_dir + '/' + img_filename + '.jpeg'

        if os.path.exists(img_path):
            image = Image.open(img_path).convert('RGB')
        else:
            img_path = self.png_dir + '/' + img_filename + '.png'
            image = Image.open(img_path).convert('RGB')

        if self.preprocess:
            processed_image = self.preprocess(image)  # Apply/run preprocessing
            # Check if preprocess returned a NumPy array or PIL Image
            if isinstance(processed_image, np.ndarray):
                image = Image.fromarray(processed_image.astype(np.uint8))  # Convert NumPy to PIL
            else:
                image = processed_image  # Already a PIL Image

        if self.transform:
            image = self.transform(image)

        return image, label

#Define normalization constants
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

#Training transformations/data augment.
train_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

#Validation transformations
val_test_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224), 
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

#Create data loaders for train/val/test sets
def get_data_loaders(train, val, test1, test2, batch_size, preprocess=None):
    train_dataset = DiabeticRetinopathyDataset(train, '/opt/anaconda3/envs/jordanenv/dr_project/aptos2019/train_images', '/opt/anaconda3/envs/jordanenv/dr_project/diabetic_retinopathy/train', transform=train_transforms, preprocess=preprocess)
    val_dataset = DiabeticRetinopathyDataset(val, '/opt/anaconda3/envs/jordanenv/dr_project/aptos2019/train_images', '/opt/anaconda3/envs/jordanenv/dr_project/diabetic_retinopathy/train', transform=val_test_transforms, preprocess=preprocess)
    test_dataset1 = DiabeticRetinopathyDataset(test1, '/opt/anaconda3/envs/jordanenv/dr_project/aptos2019/train_images', '/opt/anaconda3/envs/jordanenv/dr_project/diabetic_retinopathy/train', transform=val_test_transforms, preprocess=preprocess)
    test_dataset2 = DiabeticRetinopathyDataset(test2, '/opt/anaconda3/envs/jordanenv/dr_project/diabetic_retinopathy/train', '/opt/anaconda3/envs/jordanenv/dr_project/diabetic_retinopathy/train', transform=val_test_transforms, preprocess=preprocess)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader1 = DataLoader(test_dataset1, batch_size=batch_size, shuffle=False)
    test_loader2 = DataLoader(test_dataset2, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader1, test_loader2, test_dataset1, test_dataset2


def scale_radius(img, scale=300):
    # Estimate radius from middle row intensity
    x = img[img.shape[0] // 2, :, :].sum(1)
    r = (x > x.mean() / 10).sum() / 2
    
    # Prevent cases where radius estimation may fail
    if r < 1e-6:  # If radius is too small or zero
        r = scale / 2  # Default radius
        print(f"Warning: Radius estimation failed, using default radius {r}")
    
    s = scale * 1.0 / r #scale factor
    return cv2.resize(img, (0, 0), fx=s, fy=s)

#Preprocessing: Gaussian Subrative Normalization
def gaussian_subtractive_normalization(img, scale=300):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

    #Scale to standard radius
    img = scale_radius(img, scale)

    #Subtract local avg color
    blur = cv2.GaussianBlur(img, (0, 0), scale / 30)
    img = cv2.addWeighted(img, 4, blur, -4, 128)

    #Mask out the outer region
    mask = np.zeros(img.shape, dtype=np.uint8)
    center = (img.shape[1] // 2, img.shape[0] // 2)
    radius = int(scale * 0.9)
    cv2.circle(mask, center, radius, (1, 1, 1), -1, lineType=8)
    img = img * mask + 128 * (1 - mask)
    return img

#Preprocessing: CLAHE on Green Channel w/ Median Filter
def clahe_green_channel(img, use_median_blur=True):
    # PIL to numpy
    img = np.array(img)

    # Select Green Channel
    green_img = img[:, :, 1]

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(green_img)

    # Median Blur
    if use_median_blur:
        img_blur = cv2.medianBlur(img_clahe, 3)

    # Convert back to RGB
    img_rgb = cv2.cvtColor(img_blur, cv2.COLOR_GRAY2RGB)

    # Convert to PIL
    return Image.fromarray(img_rgb)

#Preprocessing: CLAHE w/ Gaussian Filter
def clahe_gaussian_blur(img, use_median_blur=False):
    # PIL to numpy
    img = np.array(img)

    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_gray)

    # Optional Blur (Gaussian by default)
    if use_median_blur:
        img_blur = cv2.medianBlur(img_clahe, 3)
    else:
        img_blur = cv2.GaussianBlur(img_clahe, (3, 3), 0)

    # Convert back to RGB (since the model expects 3 channels)
    img_rgb = cv2.cvtColor(img_blur, cv2.COLOR_GRAY2RGB)

    # Convert back to PIL
    return Image.fromarray(img_rgb)

#Preprocessing: Histogram Equalization and Median Filter
def hist_equalization_median_blur(img, use_median_blur=True):
    # Convert PIL to numpy
    img = np.array(img)

    # Convert to Grayscale (Histogram Equalization works on single-channel images)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply Histogram Equalization
    img_eq = cv2.equalizeHist(img_gray)

    # Apply Blur (Median by default)
    if use_median_blur:
        img_blur = cv2.medianBlur(img_eq, 3)
    else:
        img_blur = cv2.GaussianBlur(img_eq, (3, 3), 0)

    # Convert back to RGB (since the model expects 3 channels)
    img_rgb = cv2.cvtColor(img_blur, cv2.COLOR_GRAY2RGB)

    # Convert back to PIL
    return Image.fromarray(img_rgb)


def get_train_val_test_split(df):
    # Test set (20%)
    train_val_df, test_df = train_test_split(
        df, test_size=0.20, stratify=df["level"], random_state=42
    )

    # Train/val set (80%)
    train_df, val_df = train_test_split(
    train_val_df, test_size=0.125, stratify=train_val_df["level"], random_state=42
    )

    return train_df, val_df, test_df


def load_data():
    df1 = pd.read_csv("/opt/anaconda3/envs/jordanenv/dr_project/aptos2019/train.csv", dtype={"id_code": str})
    df2 = pd.read_csv("/opt/anaconda3/envs/jordanenv/dr_project/diabetic_retinopathy/trainLabels.csv", dtype={"image": str})

    # Standardize column names
    df1.rename(columns={"diagnosis": "level", "id_code": "image"}, inplace=True)

    # Clean up image IDs
    df1["image"] = df1["image"].str.strip()
    df2["image"] = df2["image"].str.strip()

    # Remove rows with missing data
    df1.dropna(subset=["image", "level"], inplace=True)
    df2.dropna(subset=["image", "level"], inplace=True)

    # Remove image names with scientific notation
    df1 = df1[~df1["image"].str.contains("E+", regex=True)]
    df2 = df2[~df2["image"].str.contains("E+", regex=True)]

    # Train/Val/Test splits
    train_df1, val_df1, test_df1 = get_train_val_test_split(df1)
    train_df2, val_df2, test_df2 = get_train_val_test_split(df2)

    # Combine & shuffle
    full_train = pd.concat([train_df1, train_df2], ignore_index=True).sample(frac=1, random_state=9).reset_index(drop=True)
    full_val = pd.concat([val_df1, val_df2], ignore_index=True).sample(frac=1, random_state=9).reset_index(drop=True)

    # Rebalance training set
    X = full_train.drop(columns=['level'])
    y = full_train['level']

    under = RandomUnderSampler(sampling_strategy={0: 5000}, random_state=42) #Undersampling to majority class 
    X_under, y_under = under.fit_resample(X, y)

    over = RandomOverSampler(sampling_strategy={label: 5000 for label in y.unique()}, random_state=42) #Oversampling to all classes
    X_balanced, y_balanced = over.fit_resample(X_under, y_under)

    full_train_balanced = pd.concat([X_balanced, y_balanced], axis=1).sample(frac=1, random_state=9).reset_index(drop=True)

    #Return balanced training, combined val, and 2 test sets
    return full_train_balanced, full_val, test_df1, test_df2




@torch.no_grad()
def evaluate(model, dataloader):
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    model.eval() #Evaluation mode
    correct = 0
    total = 0
    running_loss = 0.0

    all_preds = []
    all_labels = []

    for images, labels in tqdm(dataloader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    #Metrics
    accuracy = correct / total
    avg_loss = running_loss / len(dataloader)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    return accuracy, avg_loss, precision, recall, f1


def plot_accuracy_and_loss(train_accs, train_losses, val_accs, val_losses):
    epochs = range(1, len(train_accs) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accs, color='blue', label='Training Accuracy')
    plt.plot(epochs, val_accs, color="orange", label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(False)

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses, color='blue', label='Training Loss')
    plt.plot(epochs, val_losses, color='orange', label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(False)

    plt.tight_layout()
    plt.show()