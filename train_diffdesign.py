import torch
from torch.utils.data import DataLoader
from models.diffdesign import DiffDesign
from datasets.interior_dataset import InteriorDataset
import yaml

with open('configs/diffdesign.yaml', 'r') as f:
    config = yaml.safe_load(f)

model = DiffDesign(unet=..., tokenizer=..., meta_prior_dim=config['meta_prior_dim']).cuda()
train_set = InteriorDataset(config['data_root'], config['train_json'], random_aug=True)
train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=4)

optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
scaler = torch.cuda.amp.GradScaler()
best_loss = float('inf')

for epoch in range(config['epochs']):
    model.train()
    for batch in train_loader:
        img = batch['image'].cuda()
        prompt = batch['prompt']
        app = batch['appearance']
        spec = batch['spec']
        with torch.cuda.amp.autocast():
            loss = model(img, prompt, img, app, spec).mean()
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    print(f'Epoch {epoch} | Loss: {loss.item()}')
    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save(model.state_dict(), 'best_diffdesign.pt')
