# lpips_metric.py
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def _img_lpips_tensor(path, image_size=256):
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),  # -> [-1,1]
    ])
    return tfm(Image.open(path).convert("RGB"))


@torch.no_grad()
def lpips_score(gen_paths, gt_paths, device="cuda", net="alex", image_size=256, batch_size=16):
    import lpips
    loss_fn = lpips.LPIPS(net=net).eval().to(device)

    vals = []
    for i in range(0, len(gen_paths), batch_size):
        bg = gen_paths[i:i+batch_size]
        br = gt_paths[i:i+batch_size]
        x = torch.stack([_img_lpips_tensor(p, image_size) for p in bg]).to(device)
        y = torch.stack([_img_lpips_tensor(p, image_size) for p in br]).to(device)

        d = loss_fn(x, y).view(-1)  # (B,)
        vals.append(d.detach().cpu().numpy())

    vals = np.concatenate(vals, axis=0)
    return float(vals.mean()), float(vals.std(ddof=1))


if __name__ == "__main__":
    gen_paths = [...]
    gt_paths = [...]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m, s = lpips_score(gen_paths, gt_paths, device=device, net="alex")
    print("LPIPS mean/std:", m, s)
