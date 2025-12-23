# cmmd.py
import numpy as np
import torch


def rbf_kernel(X, Y, sigma):
    XX = (X*X).sum(dim=1, keepdim=True)
    YY = (Y*Y).sum(dim=1, keepdim=True).T
    dist2 = XX + YY - 2.0 * (X @ Y.T)
    return torch.exp(-dist2 / (2.0 * sigma * sigma))


def mmd2_unbiased_rbf(X, Y, sigma):
    n = X.shape[0]; m = Y.shape[0]
    Kxx = rbf_kernel(X, X, sigma)
    Kyy = rbf_kernel(Y, Y, sigma)
    Kxy = rbf_kernel(X, Y, sigma)
    term_xx = (Kxx.sum() - torch.diagonal(Kxx).sum()) / (n * (n - 1))
    term_yy = (Kyy.sum() - torch.diagonal(Kyy).sum()) / (m * (m - 1))
    term_xy = Kxy.mean()
    return float((term_xx + term_yy - 2.0 * term_xy).item())


def median_heuristic_sigma(Z, max_points=2000):
    if Z.shape[0] > max_points:
        idx = torch.randperm(Z.shape[0])[:max_points]
        Z = Z[idx]
    D = torch.cdist(Z, Z, p=2)
    med = torch.median(D[D > 0])
    return float(med.item() + 1e-8)


def cmmd(gen_feats, gt_feats, cond_ids, min_per_cond=2):
    """
    gen_feats, gt_feats: numpy arrays (N, d) aligned by sample index
    cond_ids: length N, each sample's condition id
    """
    gen_feats = torch.from_numpy(gen_feats).float()
    gt_feats  = torch.from_numpy(gt_feats).float()

    # global sigma
    Z = torch.cat([gen_feats, gt_feats], dim=0)
    sigma = median_heuristic_sigma(Z)

    # group indices by cond
    groups = {}
    for i, c in enumerate(cond_ids):
        groups.setdefault(c, []).append(i)

    mmds = []
    for c, idxs in groups.items():
        if len(idxs) < min_per_cond:
            continue
        X = gen_feats[idxs]
        Y = gt_feats[idxs]
        if X.shape[0] < 2 or Y.shape[0] < 2:
            continue
        mmds.append(mmd2_unbiased_rbf(X, Y, sigma))

    return float(np.mean(mmds)), float(sigma), len(mmds)


if __name__ == "__main__":
    # 你可以用 CLIP image feats 作为输入（推荐），或 Inception feats。
    gen_feats = np.load("gen_feats.npy")  # (N,d)
    gt_feats  = np.load("gt_feats.npy")   # (N,d)
    cond_ids  = [...]                     # length N

    score, sigma, used = cmmd(gen_feats, gt_feats, cond_ids)
    print("CMMD:", score, "sigma:", sigma, "conds_used:", used)
