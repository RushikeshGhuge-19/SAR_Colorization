from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os

class SAROpticalDataset(Dataset):
    def __init__(self, sar_dir, optical_dir, transform=None):
        self.sar_dir = sar_dir
        self.optical_dir = optical_dir
        self.transform = transform
        
        self.sar_images = sorted(os.listdir(sar_dir))
        self.optical_images = sorted(os.listdir(optical_dir))

    def __len__(self):
        return len(self.sar_images)

    def __getitem__(self, idx):
        sar_image_name = self.sar_images[idx]
        optical_image_name = self.optical_images[idx]

        sar_image = Image.open(os.path.join(self.sar_dir, sar_image_name)).convert('L')
        optical_image = Image.open(os.path.join(self.optical_dir, optical_image_name)).convert('RGB')

        if self.transform:
            sar_image = self.transform(sar_image)
            optical_image = self.transform(optical_image)

        return sar_image, optical_image
