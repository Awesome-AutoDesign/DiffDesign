from torch.utils.data import Dataset
from PIL import Image
import json
import os
import random

class InteriorDataset(Dataset):
    def __init__(self, data_root, json_path, transform=None, random_aug=False):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.data_root = data_root
        self.transform = transform
        self.random_aug = random_aug

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(os.path.join(self.data_root, item['image'])).convert('RGB')
        prompt = item['prompt']
        appearance = item.get('appearance', None)
        spec = item.get('spec', None)
        if self.transform:
            image = self.transform(image)
        if self.random_aug:
            if random.random() < 0.1:
                prompt += " with modern minimalism"
        return {'image': image, 'prompt': prompt, 'appearance': appearance, 'spec': spec}
