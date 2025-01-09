
import os
import random
import shutil
import torch
from models.generator import Generator
from models.discriminator import Discriminator
from data_preprocessing import SAROpticalDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision import models
from torchvision.models import VGG19_Weights
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np

# Hyperparameters
SUBSET_SIZE = 500  # Size of the image subset
BATCH_SIZE = 4     # Batch size for training
NUM_EPOCHS = 10    # Number of epochs for training
INITIAL_LR = 0.0002  # Initial learning rate
WEIGHT_PIXELWISE = 0.001  # Weight for pixel-wise loss
WEIGHT_PERCEPTUAL = 0.1     # Weight for perceptual loss
DROPOUT_RATE = 0.5  # Dropout rate for generator

# Function to create a subset of images
def create_subset(original_sar_dir, original_opt_dir, subset_size=SUBSET_SIZE):
    subset_sar_dir = "data/sar_images_subset"
    subset_opt_dir = "data/optical_images_subset"

    os.makedirs(subset_sar_dir, exist_ok=True)
    os.makedirs(subset_opt_dir, exist_ok=True)

    sar_images = os.listdir(original_sar_dir)
    opt_images = os.listdir(original_opt_dir)

    if len(sar_images) != len(opt_images):
        raise ValueError("The number of images in SAR and Optical directories must be the same.")

    selected_indices = random.sample(range(len(sar_images)), min(subset_size, len(sar_images)))

    for index in selected_indices:
        img_name = sar_images[index]
        shutil.copy(os.path.join(original_sar_dir, img_name), os.path.join(subset_sar_dir, img_name))
        img_name_opt = opt_images[index]
        shutil.copy(os.path.join(original_opt_dir, img_name_opt), os.path.join(subset_opt_dir, img_name_opt))

    print(f"Subset of {len(selected_indices)} images created successfully in '{subset_sar_dir}' and '{subset_opt_dir}'.")

# Define the original dataset directories
original_sar_dir = "data/sar"
original_opt_dir = "data/oi"

# Create a subset of images before loading the dataset
create_subset(original_sar_dir, original_opt_dir, subset_size=SUBSET_SIZE)  # Change the subset size as needed

# Initialize device, models, and loss functions
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator(dropout_rate=DROPOUT_RATE).to(device)  # Pass dropout rate to Generator
discriminator = Discriminator().to(device)

# Define loss functions
criterion_GAN = nn.MSELoss()  # Use Least Squares GAN (LSGAN) loss
criterion_pixelwise = nn.L1Loss()  # Pixel-wise loss
vgg = models.vgg19(weights=VGG19_Weights.DEFAULT).features.eval().to(device)

# Function to compute perceptual loss
def perceptual_loss(output, target):
    output_features = vgg(output)
    target_features = vgg(target)
    return nn.MSELoss()(output_features, target_features)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=INITIAL_LR, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=INITIAL_LR, betas=(0.5, 0.999))

# Define image transformations with data augmentation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),  # Randomly flip the images horizontally
    transforms.RandomRotation(15),      # Randomly rotate the images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load data from the newly created subset
dataset = SAROpticalDataset("data/sar_images_subset", "data/optical_images_subset", transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)  # Adjust batch size as needed

# Function to save generated images into a specific directory
def save_generated_images(epoch, sar_images, fake_images, save_dir='generated_images'):
    os.makedirs(save_dir, exist_ok=True)
    
    fake_images = (fake_images + 1) / 2  # Scale to [0, 1]
    plt.figure(figsize=(10, 5))
    
    # Determine the batch size dynamically
    batch_size = sar_images.size(0)
    
    for i in range(batch_size):
        plt.subplot(2, batch_size, i + 1)
        plt.imshow(sar_images[i].cpu().numpy().squeeze(), cmap='gray')
        plt.axis('off')
        plt.title("SAR Input")

        plt.subplot(2, batch_size, i + 1 + batch_size)
        plt.imshow(fake_images[i].cpu().numpy().transpose(1, 2, 0))
        plt.axis('off')
        plt.title("Generated Image")

    plt.savefig(os.path.join(save_dir, f"generated_images_epoch_{epoch}.png"))
    plt.close()

# Function to evaluate PSNR and SSIM between real and generated images
def evaluate_images(real_img, generated_img):
    # Ensure images are within the correct range [0, 1]
    real_img = np.clip(real_img, 0, 1)
    generated_img = np.clip(generated_img, 0, 1)

    # Calculate PSNR
    psnr_value = psnr(real_img, generated_img, data_range=1.0)  # Assuming images are normalized to [0, 1]

    # Calculate SSIM with a specified window size and multichannel settings
    win_size = 3  # Adjusted window size to avoid errors
    ssim_value, _ = ssim(real_img, generated_img, channel_axis=-1, win_size=win_size, full=True, data_range=1.0)

    return psnr_value, ssim_value

# Lists to store PSNR and SSIM values for each epoch
psnr_values = []
ssim_values = []

# Training loop
for epoch in range(NUM_EPOCHS):
    for i, (sar_imgs, opt_imgs) in enumerate(dataloader):
        sar_imgs, opt_imgs = sar_imgs.to(device), opt_imgs.to(device)

        # Train Discriminator
        optimizer_D.zero_grad()
        real_labels = torch.ones_like(discriminator(opt_imgs)) * 0.9  # Label smoothing
        real_loss = criterion_GAN(discriminator(opt_imgs), real_labels)
        
        fake_imgs = generator(sar_imgs)
        fake_labels = torch.zeros_like(discriminator(fake_imgs))  # Labels for fake images
        fake_loss = criterion_GAN(discriminator(fake_imgs.detach()), fake_labels)
        
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        g_loss = criterion_GAN(discriminator(fake_imgs), torch.ones_like(discriminator(fake_imgs)))  # For fake images
        g_loss += WEIGHT_PIXELWISE * criterion_pixelwise(fake_imgs, opt_imgs)  # Pixel-wise loss
        
        # Debugging: Check dimensions before computing perceptual loss
        print(f"Fake images shape: {fake_imgs.shape}, Original images shape: {opt_imgs.shape}")
        
        g_loss += WEIGHT_PERCEPTUAL * perceptual_loss(fake_imgs, opt_imgs)  # Perceptual loss
        
        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss D: {d_loss.item()}, Loss G: {g_loss.item()}")

    # Save generated images after each epoch
    with torch.no_grad():
        save_generated_images(epoch + 1, sar_imgs, fake_imgs, save_dir='generated_images')

    # Evaluate and print PSNR and SSIM for generated images
    with torch.no_grad():
        psnr_val, ssim_val = evaluate_images(opt_imgs.cpu().numpy(), fake_imgs.cpu().numpy())
        psnr_values.append(psnr_val)
        ssim_values.append(ssim_val)
        print(f"Epoch {epoch + 1}: PSNR: {psnr_val}, SSIM: {ssim_val}")

# Plot PSNR and SSIM over the epochs
plt.figure()
plt.plot(range(NUM_EPOCHS), psnr_values, label="PSNR")
plt.plot(range(NUM_EPOCHS), ssim_values, label="SSIM")
plt.xlabel("Epoch")
plt.ylabel("Metric Value")
plt.title("PSNR and SSIM over Epochs")
plt.legend()
plt.show()

print("Training Completed.")