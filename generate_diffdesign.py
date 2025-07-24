import torch
from models.diffdesign import DiffDesign
from pipelines.diffdesign_pipeline import DiffDesignPipeline

model = DiffDesign(unet=..., tokenizer=...).cuda()
model.load_state_dict(torch.load('best_diffdesign.pt'))
model.eval()

pipeline = DiffDesignPipeline(model, tokenizer=...)
prompt = "modern living room with glass wall and warm lighting"
result = pipeline(prompt, steps=50)
# Save image: e.g. torchvision.utils.save_image(result, 'out.png')
