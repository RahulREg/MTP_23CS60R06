from torchvision.io.image import read_image
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.transforms.functional import to_pil_image

from torch.utils.data import Dataset
import PIL.Image as Image
import os

from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from torch.optim import lr_scheduler
import torch

def make_dataset(root):    
    imgs = []
    
    # Paths to the RGB and GT folders
    root1 = os.path.join(root, 'rgb')
    root2 = os.path.join(root, 'gt')
    
    # List all RGB image files
    rgb_files = sorted(os.listdir(root1))
    
    for rgb_file in rgb_files:
        # Full path to the RGB image
        img = os.path.join(root1, rgb_file)
        
        # Assuming ground truth (GT) files have the same name as RGB images
        mask = os.path.join(root2, rgb_file)
        
        # Only include the pair if both the RGB and GT files exist
        if os.path.exists(mask):
            imgs.append((img, mask))
    
    return imgs

def make_test_dataset(root):
    imgs = []
    
    # Paths to the RGB and GT folders
    root1 = os.path.join(root, 'rgb')
    root2 = os.path.join(root, 'gt')
    
    # List all RGB image files
    rgb_files = sorted(os.listdir(root1))
    
    for rgb_file in rgb_files:
        # Full path to the RGB image
        img = os.path.join(root1, rgb_file)
        
        # Assuming ground truth (GT) files have the same name as RGB images
        mask = os.path.join(root2, rgb_file)
        
        # Only include the pair if both the RGB and GT files exist
        if os.path.exists(mask):
            imgs.append((img, mask))
    
    return imgs

def make_dataset_unlabeled(root):
    imgs = []
    
    # Paths to the RGB and GT folders
    root1 = os.path.join(root, 'rgb')
    # root2 = os.path.join(root, 'gt')
    
    # List all RGB image files
    rgb_files = sorted(os.listdir(root1))
    
    for rgb_file in rgb_files:
        # Full path to the RGB image
        img = os.path.join(root1, rgb_file)
        
        # Assuming ground truth (GT) files have the same name as RGB images
        # mask = os.path.join(root2, rgb_file)
        
        # Only include the pair if both the RGB and GT files exist
        if os.path.exists(img):
            imgs.append(img)
    
    return imgs

class MyDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        imgs = make_dataset(root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path)
        img_y = Image.open(y_path).convert("L")
        
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
            
        # print(torch.unique(img_y))

        # Convert target values: Map 0.0039 to 1, 0.0078 to 2, 0.0000 to 0
        # Tolerance for floating-point comparison
        tolerance = 1e-3

        # Flatten the target tensor to make replacement easier
        img_y_flat = img_y.view(-1)

        # Create masks for each target value
        mask_0 = (img_y_flat - 0.0000).abs() < tolerance
        mask_1 = (img_y_flat - 0.0039).abs() < tolerance
        mask_2 = (img_y_flat - 0.0078).abs() < tolerance

        # Replace the values using the masks
        img_y_flat[mask_0] = 0
        img_y_flat[mask_1] = 1
        img_y_flat[mask_2] = 2

        # Reshape back to the original shape
        img_y = img_y_flat.view(img_y.shape)

        return img_x, img_y

    def __len__(self):
        return len(self.imgs)

class MyDataset_unlabeled(Dataset):
    def __init__(self, root, transform=None):
        imgs = make_dataset_unlabeled(root)
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        x_path = self.imgs[index]
        img_x = Image.open(x_path)
        if self.transform is not None:
            img_x = self.transform(img_x)

        return img_x

    def __len__(self):
        return len(self.imgs)

class MyDataset_test(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        imgs = make_test_dataset(root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path)
        img_y = Image.open(y_path)
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)

        # Convert target values: Map 0.0039 to 1, 0.0078 to 2, 0.0000 to 0
        # Tolerance for floating-point comparison
        tolerance = 1e-3

        # Flatten the target tensor to make replacement easier
        img_y_flat = img_y.view(-1)

        # Create masks for each target value
        mask_0 = (img_y_flat - 0.0000).abs() < tolerance
        mask_1 = (img_y_flat - 0.0039).abs() < tolerance
        mask_2 = (img_y_flat - 0.0078).abs() < tolerance

        # Replace the values using the masks
        img_y_flat[mask_0] = 0
        img_y_flat[mask_1] = 1
        img_y_flat[mask_2] = 2

        # Reshape back to the original shape
        img_y = img_y_flat.view(img_y.shape)

        return img_x, img_y

    def __len__(self):
        return len(self.imgs)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = DeepLabV3_ResNet101_Weights.DEFAULT
model = deeplabv3_resnet101(weights=weights)
# Modify classifier head
num_classes = 3  # soil, crop, weed
model.classifier = DeepLabHead(2048, num_classes)
model.load_state_dict(torch.load('outputs/deeplabv3_resnet101_pre.pth', weights_only=True))
model = model.cuda()

x_transforms = transforms.Compose([
    transforms.Resize([256,256]),
    transforms.ToTensor(),
    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

y_transforms = transforms.Compose([
    transforms.Resize([256,256]),
    transforms.ToTensor()
])

# dataset_l = MyDataset("../../sugarbeet",transform=x_transforms,target_transform=y_transforms)
dataset_l = MyDataset("../../sunflower/test",transform=x_transforms,target_transform=y_transforms)
batch_size = 5
# print(len(liver_dataset_labeled))  # Should be greater than 0

dataloaders_labeled = DataLoader(dataset_l, batch_size=batch_size, shuffle=True, num_workers=0)

# Loss function and optimizer
class_weights = torch.tensor([1, 10, 50], dtype=torch.float).cuda()

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 800  
model.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(dataloaders_labeled):
        images = images.to(device)                       # [B, 3, H, W]
        labels = labels.squeeze(1).long().to(device)     # [B, H, W], class indices
        
        
        optimizer.zero_grad()

        outputs = model(images)['out']                   # [B, 3, H, W]
        # print(torch.unique(outputs))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}], Loss: {loss.item():.4f}")

    print(f"Epoch [{epoch+1}] finished. Avg Loss: {running_loss / len(dataloaders_labeled):.4f}")
    
# Save the model state dict
torch.save(model.state_dict(), 'outputs/deeplabv3_resnet101_pre_sun.pth')
print("✅ Model state dict saved to 'deeplabv3_resnet101_3class.pth'")

# Recreate the model structure first
# model = deeplabv3_resnet101(weights=None)
# model.classifier = DeepLabHead(2048, 3)
# model.load_state_dict(torch.load('outputs/deeplabv3_resnet101_3class.pth'))
# model = model.to(device)
# model.eval()

