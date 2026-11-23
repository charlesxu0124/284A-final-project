"""Autoencoder check: encode frames of a packed episode, decode to Gaussians, render them from the dataset cameras next to the
real images (top: real, bottom: reconstruction on white).

    python mujoco_cloth/view_recon.py --vae-ckpt runs/vae/checkpoints/latest.pth --episode data/simcloth_hold/val/episode_90.hdf5 --frames 20 80 140 --out recon.jpg
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

WS_ROOT = os.environ.get("WS_ROOT")  # a clone of https://github.com/jwang771/3d-world-simulator
if not WS_ROOT:
    raise SystemExit("set WS_ROOT to your clone of the 3D World Simulator repository")
sys.path.insert(0, WS_ROOT)
import h5video  # noqa: E402
from datasets.aloha_packed import AlohaPackedPCDataset  # noqa: E402
from models.build import load_vae, normalize_latent  # noqa: E402
from models.world_model import BigHeadRoundShim, class_masks, render_gaussians, select_class_seeds  # noqa: E402


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vae-ckpt", required=True)
    p.add_argument("--episode", required=True)
    p.add_argument("--frames", type=int, nargs="+", default=[20, 80, 140])
    p.add_argument("--out", required=True)
    a = p.parse_args()
    dev = "cuda"
    ck = torch.load(a.vae_ckpt, map_location="cpu", weights_only=False)
    vae = load_vae(ck, dev)
    feats = {}
    vae.decoder.up_finest.register_forward_hook(lambda m, i, o: feats.__setitem__("f", o))
    spec = ck["new_head"]
    head = BigHeadRoundShim(feat_dim=64, K=spec["children"], shape=spec["shape"], cap=spec["cap"]).to(dev).eval()
    head.load_state_dict(ck["seed_head_state_dict"])
    bmin, bmax = vae.bounds_min_world, vae.bounds_max_world
    vsize = (bmax - bmin) / torch.tensor(vae.grid_shapes[-1], device=dev, dtype=torch.float32)
    pm, ps = np.asarray(ck["pos_mean"], np.float32), np.asarray(ck["pos_std"], np.float32)
    ep = Path(a.episode)
    ds = AlohaPackedPCDataset(str(ep.parent), n_points=None, use_sor_filter=False)
    fi = [Path(f).name for f in ds.files].index(ep.name)
    rows = []
    with h5video.File(str(ep), "r") as f:
        cams = [int(c) for c in f.attrs["cam_ids"]]
        for t in a.frames:
            s = ds[ds.index.index((fi, t))]
            xyz = torch.from_numpy(((s["xyz"].numpy() - pm) / ps)[None]).to(dev)
            mean, _ = vae.encode(xyz, s["rgb"][None].to(dev), pos_mean=pm, pos_std=ps)
            z = normalize_latent(mean, vae.latent_split)
            out = vae.decoder(vae.conv_decoder(z), occ_mask_0=None, use_gt_mask=False)
            seeds, cls = select_class_seeds(class_masks(out["scene_occ_2"].float(), out["tac_occ_2"].float(), out["signal_2"].float(), cloth_dilate=1))
            _, r = head(feats["f"].float(), seeds, bmin, vsize)
            gt, rec = [], []
            for c in cams:
                g = f[f"render/camera_{c}"]
                im = cv2.imdecode(np.asarray(g["jpg"][t], np.uint8), cv2.IMREAD_COLOR)
                h, w = im.shape[:2]
                K, V = torch.from_numpy(g["K"][t]).to(dev), torch.from_numpy(g["viewmat"][t]).to(dev)
                fg, alpha = render_gaussians(r["xyz"].reshape(-1, 3), r["scale"].reshape(-1, 3), r["opacity"].reshape(-1), ((r["rgb"].clamp(-1, 1) + 1) / 2).reshape(-1, 3), V[None], K[None], w, h, quats=r["quat"].reshape(-1, 4))
                img = (fg[0] + (1 - alpha[0])).clamp(0, 1).mul(255).byte().cpu().numpy()[..., ::-1]
                gt.append(cv2.resize(im, (w // 2, h // 2)))
                rec.append(cv2.resize(np.ascontiguousarray(img), (w // 2, h // 2)))
            rows += [np.concatenate(gt, 1), np.concatenate(rec, 1)]
    cv2.imwrite(a.out, np.concatenate(rows, 0))
    print(f"epoch {ck.get('epoch')}: {a.out}")


if __name__ == "__main__":
    main()
