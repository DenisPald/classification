import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision import models
import yaml


def build_model(model_name: str, num_classes: int):
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, num_classes
        )

    elif model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "mobilenet_v3":
        model = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
        model.classifier[3] = nn.Linear(
            model.classifier[3].in_features, num_classes
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model.to(device)


def predict_image(image: Image.Image, model, class_names, device):
    model.eval()
    
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    return {class_names[i]: float(probs[i]) for i in range(len(class_names))}


def infer(image):
    if image is None:
        return {}

    return predict_image(image, model, class_names, device)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("DEVICE:", device)

class_names = ['Ardea_alba_egretta', 'Ardea_cocoi', 'Ardea_herodias', 'Ardea_ibis', 'Ardea_melanocephala', 'Ardea_purpurea', 'Ardeola']

with open("params.yaml") as f:
    params = yaml.safe_load(f)
cfg = params["train"]

model = build_model(model_name=cfg["model"], num_classes=len(class_names))
model.load_state_dict(torch.load("models/best_model.pt", map_location=device))
model.to(device)

demo = gr.Interface(
    fn=infer,
    inputs=gr.Image(type="pil", label="Upload image"),
    outputs=gr.Label(num_top_classes=5, label="Prediction"),
    title="Image Classification",
    description="Upload an image and the model will predict the class",
)


if __name__ == "__main__":
    demo.launch()
