# train.py
import mlflow
import mlflow.pytorch
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data.dataset import MedicalDataset
from models.unet import UNet
from utils.metrics import dice_coefficient, iou
import config

def train():
    model = UNet()
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    train_loader = DataLoader(MedicalDataset("data/train/images", "data/train/masks", config.IMG_SIZE), batch_size=config.BATCH_SIZE)

    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run():
        for epoch in range(config.EPOCHS):
            model.train()
            total_loss, total_dice, total_iou = 0, 0, 0

            for images, masks in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_dice += dice_coefficient(outputs, masks)
                total_iou += iou(outputs, masks)

            # Log metrics to MLflow
            avg_loss = total_loss / len(train_loader)
            avg_dice = total_dice / len(train_loader)
            avg_iou = total_iou / len(train_loader)
            mlflow.log_metrics({"avg_loss": avg_loss, "avg_dice": avg_dice, "avg_iou": avg_iou}, step=epoch)

            print(f"Epoch {epoch+1}/{config.EPOCHS}, Loss: {avg_loss}, Dice: {avg_dice}, IoU: {avg_iou}")

        # Save model
        mlflow.pytorch.log_model(model, config.MODEL_NAME)

if __name__ == "__main__":
    train()
