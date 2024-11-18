# app.py
import gradio as gr
import torch
from PIL import Image
from torchvision import transforms
from models.unet import UNet
import config

# Load the trained model
model = UNet()
model.load_state_dict(torch.load("models/unet.pth"))
model.eval()

def predict(img):
    transform = transforms.Compose([transforms.Resize(config.IMG_SIZE), transforms.ToTensor()])
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img)
    output = output.squeeze().numpy()
    output = (output > 0.5).astype("uint8")  # Binarize the output
    return output

gr.Interface(
    fn=predict,
    inputs=gr.inputs.Image(type="pil"),
    outputs=gr.outputs.Image(type="numpy"),
    title="Medical Image Segmentation",
    description="Upload a medical image for segmentation."
).launch()
