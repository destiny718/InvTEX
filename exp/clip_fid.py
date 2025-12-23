# clip_fid.py
import numpy as np
import torch
from PIL import Image


# ---------- CLIP image feature extractor ----------
def load_clip_model(device="cuda"):
    # 推荐 open_clip；没有的话你可以换成 transformers 版本
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model.eval().to(device)
    return model, preprocess


@torch.no_grad()
def extract_clip_image_feats(paths, device="cuda", batch_size=64):
    model, preprocess = load_clip_model(device)
    feats = []
    for i in range(0, len(paths), batch_size):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i+batch_size]]
        x = torch.stack([preprocess(im) for im in imgs]).to(device)
        f = model.encode_image(x)  # (B,dim)
        f = f / (f.norm(dim=-1, keepdim=True) + 1e-8)  # normalize
        feats.append(f.detach().cpu().numpy())
    return np.concatenate(feats, axis=0)  # (N,dim)


# ---------- FID core ----------
def mean_cov(X):
    mu = X.mean(axis=0)
    sigma = np.cov(X, rowvar=False)
    return mu, sigma


def frechet_distance(mu1, s1, mu2, s2, eps=1e-6):
    import scipy.linalg
    mu1 = np.atleast_1d(mu1); mu2 = np.atleast_1d(mu2)
    s1 = np.atleast_2d(s1);   s2 = np.atleast_2d(s2)

    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(s1 @ s2, disp=False)
    if not np.isfinite(covmean).all():
        covmean = scipy.linalg.sqrtm((s1 + eps*np.eye(s1.shape[0])) @ (s2 + eps*np.eye(s2.shape[0])))

    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(s1) + np.trace(s2) - 2.0 * np.trace(covmean))


def clip_fid(gen_feats, gt_feats):
    mu_g, s_g = mean_cov(gen_feats)
    mu_r, s_r = mean_cov(gt_feats)
    return frechet_distance(mu_g, s_g, mu_r, s_r)


if __name__ == "__main__":
    gen_paths = [...]
    gt_paths = [...]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    g = extract_clip_image_feats(gen_paths, device=device)
    r = extract_clip_image_feats(gt_paths, device=device)
    print("CLIP-FiD:", clip_fid(g, r))
