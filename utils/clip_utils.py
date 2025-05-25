import clip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

def get_image_embedding(image):
    image_input = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
    with torch.no_grad():
        return clip_model.encode_image(image_input).float().squeeze()


def get_text_embedding(text):
    text_input = clip.tokenize([text]).to(device)
    with torch.no_grad():
        return clip_model.encode_text(text_input).float().squeeze()
