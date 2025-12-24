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
    gen_paths = ["/data/tianqi/InvTEX/data/hy/hunyuan_case1.png",
                "/data/tianqi/InvTEX/data/hy/hunyuan_case2.png",
                "/data/tianqi/InvTEX/data/hy/hunyuan_case3.png",
                "/data/tianqi/InvTEX/data/hy/hunyuan_case4.png"]
    prompts = ["The head features a typical tiger stripe pattern, primarily orange-yellow, with black stripes and creamy white areas around the mouth and nose. The eyes are large and bright, and the tip of the nose is pink. The body and limbs are covered with metallic plates in dark gray and gunmetal, with brown leather straps, and are accented with antique bronze details in certain areas.", "a red bag filled with colorful gift boxes. The bag is slightly rounded, and the gift boxes inside appear to have different sizes and colorful wrapping, with bows on some of them. The scene has a simple, clean look with a minimalistic background.", "Stylized human figure with asymmetrical pinkish-purple hair, green and brown rolled-up sleeve shirt with checkerboard collar and spike details, half beige, half purple quilted knee-length skirt, short brown vest, turquoise calf-height boots with pink details, and multiple black, grey, and metallic bracelets.", "a small, vibrant urban scene with two buildings, a road, and some street elements. The buildings have various signs, plants, and a large cat sculpture on the rooftop. The street has utility poles, some construction cones, and a cozy, detailed environment with a mix of architectural styles. The scene has a playful, cartoonish aesthetic."]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m, s = clip_t(gen_paths, prompts, device=device)
    print("CLIP-T mean/std:", m, s)


