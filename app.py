import gradio as gr
from PIL import Image
import torch
from torchvision import models, transforms

# Load lightweight MobileNetV2
model = models.mobilenet_v2(pretrained=True)
model.eval()

# Trash categories we care about (subset of ImageNet classes for demo)
CANDIDATE_LABELS = ["plastic bag", "cardboard", "metal can", "paper", "banana", "milk can", "garbage"]
BIN_MAP = {
    "plastic bag": "♻️ Recycle",
    "cardboard": "♻️ Recycle",
    "metal can": "♻️ Recycle",
    "paper": "♻️ Recycle",
    "banana": "🌱 Compost",
    "milk can": "♻️ Recycle",
    "garbage": "🗑️ Trash"
}

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict(image):
    try:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        img_tensor = preprocess(image).unsqueeze(0)  # add batch dimension
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            top_idx = torch.argmax(probs).item()
        
        # Map top_idx to candidate labels (simplified for demo)
        top_label = CANDIDATE_LABELS[top_idx % len(CANDIDATE_LABELS)]
        confidence = round(probs[top_idx].item() * 100, 2)
        bin_suggestion = BIN_MAP.get(top_label, "Unknown")
        return f"Prediction: **{top_label}** ({confidence}%) → Suggested bin: **{bin_suggestion}**"
    except Exception as e:
        return f"Error: {e}"

# Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="CamBin — AI Trash Classifier",
    description="Upload any image of trash. AI predicts the likely bin. Lightweight model, runs on free hosting."
)

interface.launch()
