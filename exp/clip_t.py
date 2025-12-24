# clip_t.py
import numpy as np
import torch
from PIL import Image


def load_clip(device="cuda"):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval().to(device)
    return model, preprocess, tokenizer


@torch.no_grad()
def clip_t(gen_paths, prompts, device="cuda", batch_size=64):
    model, preprocess, tokenizer = load_clip(device)
    scores = []

    for i in range(0, len(gen_paths), batch_size):
        bp = gen_paths[i:i+batch_size]
        bt = prompts[i:i+batch_size]

        imgs = [Image.open(p).convert("RGB") for p in bp]
        x = torch.stack([preprocess(im) for im in imgs]).to(device)
        t = tokenizer(bt).to(device)

        img_f = model.encode_image(x)
        txt_f = model.encode_text(t)
        img_f = img_f / (img_f.norm(dim=-1, keepdim=True) + 1e-8)
        txt_f = txt_f / (txt_f.norm(dim=-1, keepdim=True) + 1e-8)

        sim = (img_f * txt_f).sum(dim=-1)  # cosine
        scores.append(sim.detach().cpu().numpy())

    scores = np.concatenate(scores, axis=0)
    return float(scores.mean()), float(scores.std(ddof=1))


if __name__ == "__main__":
    gen_paths = ["baked_texture_new.png"]
    prompts = ["A stylized, colorful underwater coral reef with a geometric grey rock base featuring a green cactus-like coral at the top, beige branching coral, orange tube coral, red pebble coral, green polyp coral, yellow fan coral, light purple lichen coral, pink coral, blue anemones, red coral, and blue bubble coral."]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m, s = clip_t(gen_paths, prompts, device=device)
    print("CLIP-T mean/std:", m, s)
