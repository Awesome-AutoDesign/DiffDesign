import torch

class DiffDesignPipeline:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def __call__(self, prompt, image=None, appearance=None, spec=None, mask=None, steps=50, return_intermediate=False):
        x = torch.randn(1, self.model.unet.input_blocks[0][0].in_channels, 256, 256).to(next(self.model.parameters()).device)
        outs = []
        for i in range(steps):
            x = self.model(x, prompt, image, appearance, spec, mask)
            if return_intermediate and (i+1) % 10 == 0:
                outs.append(x.clone())
        return x if not return_intermediate else outs

    def grid_sample(self, prompts, imgs, steps=20):
        results = []
        for p, img in zip(prompts, imgs):
            results.append(self.__call__(p, image=img, steps=steps))
        return results
