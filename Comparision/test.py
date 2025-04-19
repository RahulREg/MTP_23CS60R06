
import os
import torch
from PIL import Image
from matplotlib import pyplot as plt

import models.segnet as models
# import models.erfnet as models

from torchvision.io.image import read_image
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.transforms.functional import to_pil_image

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
import numpy as np
import cv2

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
        
        # rgb_image = Image.open(x_path).convert("RGB")
        # gt_image = Image.open(y_path).convert("L")  # Load as grayscale "L" mode

        # if self.transform:
        #     rgb_image = self.transform(rgb_image)
        #     # Clip values to [0, 1] to avoid imshow warnings
        #     # rgb_image = np.clip(rgb_image, 0, 1)
        #     gt_image = gt_image.resize((256, 256), Image.NEAREST)
        #     # Ensure gt_image maintains integer labels
        #     gt_image = np.array(gt_image)  # Convert to numpy array

        #     # Map values: 0 -> 0, 2 -> 1, 255 -> 2
        #     # gt_image[gt_image == 0] = 0
        #     # gt_image[gt_image == 2] = 1
        #     # gt_image[gt_image == 255] = 2

        #     gt_image = torch.from_numpy(gt_image).long()  # Convert to PyTorch tensor

        # # Convert to numpy array for processing
        # # rgb_image_np = np.array(rgb_image.permute(1, 2, 0))  # Change from (C, H, W) to (H, W, C)
        # rgb_image_np = rgb_image.permute(1, 2, 0).numpy()


        # # Split channels
        # I1, I2, I3 = rgb_image_np[:, :, 0], rgb_image_np[:, :, 1], rgb_image_np[:, :, 2]

        # # Calculate additional channels
        # IExG = 2 * I2 - I1 - I3
        # IExR = 1.4 * I1 - I2
        # ICIVE = 0.881 * I2 - 0.441 * I1 - 0.385 * I3 - 18.78745
        # INDI = (I2 - I1) / (I2 + I1 + 1e-6)  # Adding small value to avoid division by zero

        # hsv_image = cv2.cvtColor(rgb_image_np, cv2.COLOR_RGB2HSV)
        # IHUE, ISAT, IVAL = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]

        # # Convert to a type compatible with OpenCV functions
        # IExG = IExG.astype(np.float32)
        # rxIExG = cv2.Sobel(IExG, cv2.CV_32F, 1, 0, ksize=3)
        # ryIExG = cv2.Sobel(IExG, cv2.CV_32F, 0, 1, ksize=3)
        # r2IExG = cv2.Laplacian(IExG, cv2.CV_32F)
        # IEDGES = cv2.Canny((IExG * 255).astype(np.uint8), 100, 200)

        # # Stack all channels together
        # additional_channels = np.stack([IExG, IExR, ICIVE, INDI, IHUE, ISAT, IVAL, rxIExG, ryIExG, r2IExG, IEDGES], axis=2)
        # additional_channels = torch.from_numpy(additional_channels).permute(2, 0, 1)  # Change to (C, H, W)

        # rgb_image = rgb_image.float()
        # additional_channels = additional_channels.float()

        # # Concatenate RGB and additional channels
        # final_image = torch.cat([rgb_image, additional_channels], dim=0)

        # return final_image, gt_image


    def __len__(self):
        return len(self.imgs)

def compute_iou(conf_matrix):
    intersection = np.diag(conf_matrix)
    union = np.sum(conf_matrix, axis=1) + np.sum(conf_matrix, axis=0) - intersection
    iou = intersection / np.maximum(union, 1)
    return iou

def evaluate_model(model, dataloader, device, num_classes=3):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            outputs = outputs['out'] if isinstance(outputs, dict) else outputs
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(targets.cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    iou = compute_iou(cm)

    return acc, prec, rec, iou.mean(), iou


# Function to load the student and teacher models
def load_model(model_path):
    # student_model = DenseUnet_2d().cuda()
    
    # student_model = Unet(3,3).cuda()

    # student_model = models.SegNet(14,3).cuda()

    # student_model = models.ERFNet(3).cuda()
    
    weights = DeepLabV3_ResNet101_Weights.DEFAULT
    student_model = deeplabv3_resnet101(weights=weights)
    # Modify classifier head
    num_classes = 3  # soil, crop, weed
    student_model.classifier = DeepLabHead(2048, num_classes)
    student_model = student_model.cuda()
    
    # 2. Load the checkpoint
    checkpoint = torch.load(model_path, weights_only=True)
    
    # 3. Load the model weights
    student_model.load_state_dict(checkpoint)
    
    # Set models to evaluation mode
    student_model.eval()
    
    print(f"Loaded models from {student_path}")
        
    return student_model

# Function to predict a mask for a given image
def predict_image(model, image_path, transform):
    # Load and preprocess the input image
    # img = Image.open(image_path)
    # img = transform(img).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')  # Add batch dimension
    
    import torch
    from PIL import Image
    import numpy as np

    img = Image.open(image_path).convert("RGB").resize((256, 256))  # resize manually
    img = np.array(img, dtype=np.float32) / 255.0  # manually divide by 255
    img = torch.from_numpy(img).permute(2, 0, 1)  # convert to [C, H, W]
    
    # rgb_image = Image.open(image_path).convert("RGB")

    # rgb_image = transform(rgb_image)

    # rgb_image_np = rgb_image.permute(1, 2, 0).numpy()


    # # Split channels
    # I1, I2, I3 = rgb_image_np[:, :, 0], rgb_image_np[:, :, 1], rgb_image_np[:, :, 2]

    # # Calculate additional channels
    # IExG = 2 * I2 - I1 - I3
    # IExR = 1.4 * I1 - I2
    # ICIVE = 0.881 * I2 - 0.441 * I1 - 0.385 * I3 - 18.78745
    # INDI = (I2 - I1) / (I2 + I1 + 1e-6)  # Adding small value to avoid division by zero

    # hsv_image = cv2.cvtColor(rgb_image_np, cv2.COLOR_RGB2HSV)
    # IHUE, ISAT, IVAL = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]

    # # Convert to a type compatible with OpenCV functions
    # IExG = IExG.astype(np.float32)
    # rxIExG = cv2.Sobel(IExG, cv2.CV_32F, 1, 0, ksize=3)
    # ryIExG = cv2.Sobel(IExG, cv2.CV_32F, 0, 1, ksize=3)
    # r2IExG = cv2.Laplacian(IExG, cv2.CV_32F)
    # IEDGES = cv2.Canny((IExG * 255).astype(np.uint8), 100, 200)

    # # Stack all channels together
    # additional_channels = np.stack([IExG, IExR, ICIVE, INDI, IHUE, ISAT, IVAL, rxIExG, ryIExG, r2IExG, IEDGES], axis=2)
    # additional_channels = torch.from_numpy(additional_channels).permute(2, 0, 1)  # Change to (C, H, W)

    # rgb_image = rgb_image.float()
    # additional_channels = additional_channels.float()

    # # Concatenate RGB and additional channels
    # img = torch.cat([rgb_image, additional_channels], dim=0)
    
    
    img = img.unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')


    
    # Predict the mask
    with torch.no_grad():
        output = model(img)
        output = output['out'] if isinstance(output, dict) else output
        predicted_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()  # Convert to class predictions
    return predicted_mask

if __name__ == '__main__':    
    # Paths to dataset
    # rgb_folder = "../../../Data/test/rgb"
    # gt_folder = "../../../Data/test/gt"
    
    rgb_folder = "../../sunflower/val/rgb"
    gt_folder = "../../sunflower/val/gt"
    
    # Load models
    student_path = 'outputs/deeplabv3_resnet101_pre_sun.pth'
    student_model = load_model(student_path)
    
    output_folder = "visualization/sun_dlv3_pre"
    
    # Define the image transformation
    from torchvision.transforms import transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Loop through all images in the folder
    import os

    # Define output folder for saving plots
    # output_folder = "visualization/sugar"
    
    os.makedirs(output_folder, exist_ok=True)  # Create it if it doesn't exist

    for image_name in os.listdir(rgb_folder):
        # Skip non-image files
        if not image_name.endswith('.png'):
            continue

        # Paths for input image and ground truth mask
        image_path = os.path.join(rgb_folder, image_name)
        gt_path = os.path.join(gt_folder, image_name)

        # Predict the mask
        prediction = predict_image(student_model, image_path, transform)
        print("Unique predicted values:", np.unique(prediction))

        # Load the ground truth mask
        gt_mask = Image.open(gt_path).convert("L").resize((256, 256))
        gt_mask = np.array(gt_mask, dtype=np.float32)

        # Display the image, ground truth, and prediction
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(Image.open(image_path))
        axes[0].set_title("Input Image")
        axes[0].axis('off')

        axes[1].imshow(gt_mask, cmap='gray')
        axes[1].set_title("Ground Truth")
        axes[1].axis('off')

        axes[2].imshow(prediction, cmap='gray')
        axes[2].set_title("Predicted Mask")
        axes[2].axis('off')

        plt.tight_layout()

        # Save the figure
        save_path = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_comparison.png")
        plt.savefig(save_path)
        plt.close(fig)  # Close the figure to avoid memory buildu


    x_transforms = transforms.Compose([
        transforms.Resize([256,256]),
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    y_transforms = transforms.Compose([
        transforms.Resize([256,256]),
        transforms.ToTensor()
    ])
    
    dataset_l = MyDataset("../../sunflower/val",transform=x_transforms,target_transform=y_transforms)
    batch_size = 10
    # print(len(liver_dataset_labeled))  # Should be greater than 0

    val_loader = DataLoader(dataset_l, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # assuming val_loader is your validation dataloader
    student_metrics = evaluate_model(student_model, val_loader, device)

    # Unpack
    s_acc, s_prec, s_rec, s_miou, s_iou_per_class = student_metrics

    # Print
    print("📘 Model:")
    print(f"Accuracy: {s_acc:.4f}")
    print(f"Precision: {s_prec:.4f}")
    print(f"Recall: {s_rec:.4f}")
    print(f"Mean IoU: {s_miou:.4f}")
    print(f"Per-class IoU: Soil: {s_iou_per_class[0]:.4f}, Crop: {s_iou_per_class[1]:.4f}, Weed: {s_iou_per_class[2]:.4f}\n")