import torch
from transformers import CLIPModel, CLIPProcessor

class CLIPAdapter:
    def __init__(self, model_name):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device).eval()
        self.cache = {}

    @torch.no_grad()
    def get_text_features(self, prompt):
        key = f"text:{prompt}"
        if key in self.cache:
            return self.cache[key]
        inputs = self.processor(text=prompt, return_tensors="pt", padding=True).to(self.device)
        feat = self.model.get_text_features(**inputs)
        self.cache[key] = feat
        return feat

    @torch.no_grad()
    def get_image_features(self, images):
        key = f"img:{id(images)}"
        if key in self.cache:
            return self.cache[key]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        feat = self.model.get_image_features(**inputs)
        self.cache[key] = feat
        return feat
