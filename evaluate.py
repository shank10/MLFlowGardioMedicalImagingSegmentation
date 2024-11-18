# evaluate.py
import torch
from data.dataset import MedicalDataset
from models.unet import UNet
from utils.metrics import dice_coefficient, iou
from torch.utils.data import DataLoader
import config

def evaluate():
    model = UNet()
    model.load_state_dict(torch.load("models/unet.pth"))
    model.eval()
    val_loader = DataLoader(MedicalDataset("data/val/images", "data/val/masks", config.IMG_SIZE), batch_size=config.BATCH_SIZE)

    dice, iou_score = 0, 0
    with torch.no_grad():
        for images, masks in val_loader:
            outputs = model(images)
            dice += dice_coefficient(outputs, masks)
            iou_score += iou(outputs, masks)
    print(f"Dice Coefficient: {dice/len(val_loader)}, IoU: {iou_score/len(val_loader)}")

if __name__ == "__main__":
    evaluate()
