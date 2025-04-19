import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import models.segnet as models
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

# Custom dataset class
class SunflowerDataset(Dataset):
    def __init__(self, rgb_dir, gt_dir, transform=None):
        self.rgb_dir = rgb_dir
        self.gt_dir = gt_dir
        self.rgb_images = os.listdir(rgb_dir)
        self.transform = transform

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, idx):
        while True:
            rgb_image_path = os.path.join(self.rgb_dir, self.rgb_images[idx])
            gt_image_path = os.path.join(self.gt_dir, self.rgb_images[idx])

            if not os.path.exists(gt_image_path):
                idx = (idx + 1) % len(self.rgb_images)
                continue

            rgb_image = Image.open(rgb_image_path).convert("RGB")
            gt_image = Image.open(gt_image_path).convert("L")  # Load as grayscale "L" mode

            if self.transform:
                rgb_image = self.transform(rgb_image)
                # Clip values to [0, 1] to avoid imshow warnings
                # rgb_image = np.clip(rgb_image, 0, 1)
                gt_image = gt_image.resize((256, 256), Image.NEAREST)
                # Ensure gt_image maintains integer labels
                gt_image = np.array(gt_image)  # Convert to numpy array

                # Map values: 0 -> 0, 2 -> 1, 255 -> 2
                # gt_image[gt_image == 0] = 0
                # gt_image[gt_image == 2] = 1
                # gt_image[gt_image == 255] = 2

                gt_image = torch.from_numpy(gt_image).long()  # Convert to PyTorch tensor

            # Convert to numpy array for processing
            # rgb_image_np = np.array(rgb_image.permute(1, 2, 0))  # Change from (C, H, W) to (H, W, C)
            rgb_image_np = rgb_image.permute(1, 2, 0).numpy()


            # Split channels
            I1, I2, I3 = rgb_image_np[:, :, 0], rgb_image_np[:, :, 1], rgb_image_np[:, :, 2]

            # Calculate additional channels
            IExG = 2 * I2 - I1 - I3
            IExR = 1.4 * I1 - I2
            ICIVE = 0.881 * I2 - 0.441 * I1 - 0.385 * I3 - 18.78745
            INDI = (I2 - I1) / (I2 + I1 + 1e-6)  # Adding small value to avoid division by zero

            hsv_image = cv2.cvtColor(rgb_image_np, cv2.COLOR_RGB2HSV)
            IHUE, ISAT, IVAL = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]

            # Convert to a type compatible with OpenCV functions
            IExG = IExG.astype(np.float32)
            rxIExG = cv2.Sobel(IExG, cv2.CV_32F, 1, 0, ksize=3)
            ryIExG = cv2.Sobel(IExG, cv2.CV_32F, 0, 1, ksize=3)
            r2IExG = cv2.Laplacian(IExG, cv2.CV_32F)
            IEDGES = cv2.Canny((IExG * 255).astype(np.uint8), 100, 200)

            # Stack all channels together
            additional_channels = np.stack([IExG, IExR, ICIVE, INDI, IHUE, ISAT, IVAL, rxIExG, ryIExG, r2IExG, IEDGES], axis=2)
            additional_channels = torch.from_numpy(additional_channels).permute(2, 0, 1)  # Change to (C, H, W)

            rgb_image = rgb_image.float()
            additional_channels = additional_channels.float()

            # Concatenate RGB and additional channels
            final_image = torch.cat([rgb_image, additional_channels], dim=0)

            return final_image, gt_image

# Define paths
rgb_dir = '../../sunflower/test/rgb'
gt_dir = '../../sunflower/test/gt'

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create dataset
dataset = SunflowerDataset(rgb_dir, gt_dir, transform=transform)

# Create dataloader
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Get one example
final_image, gt_image = next(iter(dataloader))

# Print the shapes for debugging
print(final_image.shape, gt_image.shape)

# Find min and max values in ground truth image
min_val = torch.min(gt_image)
max_val = torch.max(gt_image)
print(f"Min value in ground truth image: {min_val}")
print(f"Max value in ground truth image: {max_val}")

# Find unique values in the ground truth image
unique_values = torch.unique(gt_image)
print(f"Unique values in ground truth image: {unique_values}")

# Convert tensors to numpy arrays
# Note: The first three channels are RGB
rgb_image_np = final_image.squeeze(0)[:3].numpy().transpose((1, 2, 0))
gt_image_np = gt_image.numpy().squeeze()  # Squeeze to remove the channel dimension for grayscale

# # Plot the images
# fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# ax[0].imshow(rgb_image_np)
# ax[0].set_title("RGB Image")
# ax[0].axis('off')

# ax[1].imshow(gt_image_np, cmap='gray')  # Use grayscale colormap for gt_image
# ax[1].set_title("Ground Truth Image (Grayscale)")
# ax[1].axis('off')

# plt.show()


# Initialize the model
model = models.SegNet(14, 3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

model.load_state_dict(torch.load('outputs/segnet_pre_sugar.pth', weights_only=True))

# Hyperparameters
lr = 1e-4
w_decay = 0
n_epochs = 800
batch_size = 2  # Adjust based on your GPU memory
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# sigma_scale = 11.0
# alpha_scale = 11.0

# Define loss and optimizer
class_weights = torch.tensor([1, 10, 50], dtype=torch.float).cuda()

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=w_decay)

# Training loop
for epoch in range(n_epochs):
    running_loss = 0.0
    print('Epoch:', epoch + 1)
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device).long()
        # print("Unique values in labels:", torch.unique(labels))
        # print(labels.shape)
        # print(labels)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        # print(outputs.shape)
        loss = criterion(outputs, labels.squeeze(1))
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 10 == 9:  # Print every 500 mini-batches
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 10}")
            running_loss = 0.0

    # Optionally, you can add validation accuracy calculation here
torch.save(model.state_dict(), 'outputs/segnet_pre.pth')
print('Finished Training')
