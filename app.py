import gradio as gr
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

# Load the CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Labels and corresponding bins
CANDIDATE_LABELS = ["Plastic", "Cardboard", "Metal", "Paper", "Food", "Trash", "Compost"]
BIN_MAP = {
    "Plastic": "♻️ Recycle",
    "Cardboard": "♻️ Recycle",
    "Metal": "♻️ Recycle",
    "Paper": "♻️ Recycle",
    "Food": "🌱 Compost",
    "Trash": "🗑️ Trash",
    "Compost": "🌱 Compost"
}

def predict(image):
    try:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # Use CLIP to score labels
        inputs = processor(text=CANDIDATE_LABELS, images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).squeeze()
        top_idx = torch.argmax(probs).item()
        top_label = CANDIDATE_LABELS[top_idx]
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
    description="Upload any image of trash. AI predicts the likely bin. Fully AI-powered; no hard-coded demo items."
)

interface.launch()