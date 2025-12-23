import torch
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import numpy as np
import torch.nn as nn
import yaml
import chromadb
from chromadb.utils import embedding_functions

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("DEVICE:", device)


image_dir = Path("data/images/processed/train")
db_path = Path("embeddings/chroma_db") 
db_path.mkdir(parents=True, exist_ok=True)

class_names = ['Ardea_alba_egretta', 'Ardea_cocoi', 'Ardea_herodias', 'Ardea_ibis', 'Ardea_melanocephala', 'Ardea_purpurea', 'Ardeola']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def build_model(model_name: str, num_classes: int):
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "mobilenet_v3":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model

with open("params.yaml") as f:
    params = yaml.safe_load(f)
cfg = params["train"]

model = build_model(model_name=cfg["model"], num_classes=len(class_names))
model.load_state_dict(torch.load("models/best_model.pt", map_location=device))

feature_extractor = nn.Sequential(
    *list(model.children())[:-1],
    nn.Flatten()
)
feature_extractor.to(device)
feature_extractor.eval()

client = chromadb.PersistentClient(path=str(db_path))
collection_name = "bird_embeddings"

if collection_name in [c.name for c in client.list_collections()]:
    client.delete_collection(collection_name)
collection = client.create_collection(name=collection_name)

for class_folder in image_dir.iterdir():
    if not class_folder.is_dir():
        continue
    
    for img_path in class_folder.glob("*"):
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
            
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            emb = feature_extractor(x).squeeze().cpu().numpy()

        collection.add(
            embeddings=[emb.tolist()],
            metadatas=[{"class_name": class_folder.name, "path": str(img_path)}],
            ids=[str(img_path.name)]
        )

print(f"Успешно! Эмбеддинги сохранены в {db_path}")