# kid.py
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import inception_v3


# ---------- feature extractor ----------
def _img_to_tensor(path, image_size=299):
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),  # -> [-1,1]
    ])
    return tfm(Image.open(path).convert("RGB"))


@torch.no_grad()
def extract_inception_feats(paths, device="cuda", batch_size=32, image_size=299):
    model = inception_v3(weights="IMAGENET1K_V1", aux_logits=False)
    model.fc = torch.nn.Identity()  # output 2048-d
    model.eval().to(device)

    feats = []
    for i in range(0, len(paths), batch_size):
        x = torch.stack([_img_to_tensor(p, image_size) for p in paths[i:i+batch_size]]).to(device)
        f = model(x)  # (B, 2048)
        feats.append(f.detach().cpu())
    return torch.cat(feats, dim=0)  # (N,2048)


# ---------- KID core (polynomial MMD^2 unbiased) ----------
def poly_mmd2_unbiased(X, Y, degree=3, gamma=None, coef0=1.0):
    """
    X: (n,d) torch
    Y: (m,d) torch
    k(x,y)=(gamma*x^T y + coef0)^degree
    """
    X = X.float()
    Y = Y.float()
    n, d = X.shape
    m, _ = Y.shape
    if gamma is None:
        gamma = 1.0 / d

    Kxx = (gamma * (X @ X.T) + coef0).pow(degree)
    Kyy = (gamma * (Y @ Y.T) + coef0).pow(degree)
    Kxy = (gamma * (X @ Y.T) + coef0).pow(degree)

    # unbiased: remove diagonal
    term_xx = (Kxx.sum() - torch.diagonal(Kxx).sum()) / (n * (n - 1))
    term_yy = (Kyy.sum() - torch.diagonal(Kyy).sum()) / (m * (m - 1))
    term_xy = Kxy.mean()
    return (term_xx + term_yy - 2.0 * term_xy).item()


def kid(gen_feats, gt_feats, subset_size=100, n_subsets=50, seed=0):
    """
    KID = mean over subsets of MMD^2 (poly kernel). Return (mean, std).
    """
    rng = np.random.default_rng(seed)
    N = gen_feats.shape[0]
    M = gt_feats.shape[0]
    subset_size = min(subset_size, N, M)

    scores = []
    for _ in range(n_subsets):
        ig = rng.choice(N, subset_size, replace=False)
        ir = rng.choice(M, subset_size, replace=False)
        scores.append(poly_mmd2_unbiased(gen_feats[ig], gt_feats[ir]))
    scores = np.array(scores, dtype=np.float64)
    return float(scores.mean()), float(scores.std(ddof=1))


# ---------- usage sketch ----------
if __name__ == "__main__":
    # gen_paths, gt_paths 由你自己提供
    gen_paths = [...]
    gt_paths = [...]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gf = extract_inception_feats(gen_paths, device=device)
    rf = extract_inception_feats(gt_paths, device=device)
    mean, std = kid(gf, rf, subset_size=100, n_subsets=50)
    print("KID mean/std:", mean, std)
