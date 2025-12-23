import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
import yaml
import chromadb
from pathlib import Path

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class_names = ['Ardea_alba_egretta', 'Ardea_cocoi', 'Ardea_herodias', 'Ardea_ibis', 'Ardea_melanocephala', 'Ardea_purpurea', 'Ardeola']

db_path = Path("embeddings/chroma_db")
client = chromadb.PersistentClient(path=str(db_path))
collection = client.get_collection(name="bird_embeddings")

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
    return model

with open("params.yaml") as f:
    params = yaml.safe_load(f)
cfg = params["train"]

full_model = build_model(model_name=cfg["model"], num_classes=len(class_names))
full_model.load_state_dict(torch.load("models/best_model.pt", map_location=device))
full_model.to(device).eval()

feature_extractor = nn.Sequential(
    *list(full_model.children())[:-1],
    nn.Flatten()
).to(device).eval()


def infer(image):
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = full_model(img_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        label_results = {class_names[i]: float(probs[i]) for i in range(len(class_names))}

        embedding = feature_extractor(img_tensor).squeeze().cpu().numpy()
        
        results = collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=1
        )

    if results['metadatas'] and results['metadatas'][0]:
        closest_img_path = results['metadatas'][0][0]['path']
        closest_img = Image.open(closest_img_path)
        distance = results['distances'][0][0]
        match_info = f"Изображение: {results['metadatas'][0][0]['class_name']} (Дистанция: {distance:.4f})"
    else:
        closest_img = None
        match_info = "Не найдено"

    return label_results, closest_img, match_info

with gr.Blocks(title="Классификация птиц") as demo:
    gr.Markdown("Классификация птиц")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Загрузка изображения")
            btn = gr.Button("Analyze")
        
        with gr.Column():
            output_label = gr.Label(num_top_classes=3, label="Предсказание")
            output_match_info = gr.Textbox(label="Результат векторного поиска")
            output_closest_img = gr.Image(label="Самое похожее изображение")

    btn.click(
        fn=infer, 
        inputs=input_img, 
        outputs=[output_label, output_closest_img, output_match_info]
    )

if __name__ == "__main__":
    demo.launch()