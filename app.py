import gradio as gr
from transformers import pipeline
from PIL import Image

# Category options
categories = ["Recycle", "Compost", "Trash"]

# Load models
image_model = pipeline("image-classification", model="microsoft/resnet-50")
text_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def classify_trash(image, description):
    # Get predictions
    image_pred = image_model(image)[0]["label"].lower()
    text_pred = text_model(description, candidate_labels=categories)
    text_pred_label = text_pred["labels"][0]

    # Decision logic
    if image_pred == text_pred_label.lower():
        result = f"✅ Confident: {text_pred_label}"
    else:
        result = f"🤔 Perhaps: {text_pred_label}"

    return result

# Build app
app = gr.Interface(
    fn=classify_trash,
    inputs=[
        gr.Image(type="pil", label="Take or Upload a Photo"),
        gr.Textbox(label="Describe the Item (e.g. banana peel, soda can)")
    ],
    outputs=gr.Textbox(label="Sorting Suggestion"),
    title="♻️ Cam-Bin",
    description="Upload a photo and describe the item. The AI will tell whether it's compost, recycle, or trash — showing confidence if both agree."
)

if __name__ == "__main__":
    app.launch()
