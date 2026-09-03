"""
Stage 3: two-branch 1-D CNN for transit vetting, after Shallue &
Vanderburg 2018 ("Astronet").

    global (B,1,2001) -> conv stack (5 blocks, 16..256) -> flatten -.
                                                                    |
                                                                    +-> FC 512 -> FC 512 -> logit
                                                                    |
    local  (B,1, 201) -> conv stack (2 blocks, 16.. 32) -> flatten -'

WHAT THIS IS TESTING
--------------------
Stage 0's Random Forest on catalogue summary statistics scores PR-AUC
0.947. Two of the four false-positive tells are SHAPE-based (a grazing
binary is V-shaped rather than U-shaped; a secondary eclipse sits near
phase 0.5), and a scalar summary cannot express either. The hypothesis is
that a CNN reading the curve beats the tree reading the summary.

If it does not, that is a real finding and gets reported as one.

DELIBERATE CHOICES
------------------
* Depth is NOT fed to the network. views.h5 normalises every curve to
  baseline 0, depth -1 precisely so the model cannot latch onto
  "deep = binary", which is the one cue the tree already handles
  perfectly. Feeding depth back would confound the hypothesis. It stays
  in the file for a later ablation.

* The split comes from splits.npz and is FIXED across seeds. Varying the
  seed varies initialisation, augmentation and batch order - model
  variance - while the held-out set stays the held-out set. Re-drawing
  the split per seed would conflate the two and quietly re-use test.

* Augmentation is horizontal reflection of BOTH views together. A transit
  is time-symmetric about mid-transit, so mirroring phase is
  label-preserving. The two views share one phase axis, so they must flip
  together or the pair becomes incoherent.

* pos_weight handles the 39.7/60.3 imbalance rather than resampling, so
  every epoch sees every example.

TEST SET DISCIPLINE
-------------------
Model selection is on val, by PR-AUC. Test is evaluated once per seed,
after training finishes, and never influences any choice.
"""

import argparse
import json
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

H5 = "views.h5"
SPLITS = "splits.npz"


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------

def conv_block(cin, cout):
    """[Conv(k=5) x2, MaxPool(2)]. 'same' padding keeps the arithmetic
    clean and preserves the edges of the local view, where ingress and
    egress live."""
    return nn.Sequential(
        nn.Conv1d(cin, cout, 5, padding=2), nn.ReLU(),
        nn.Conv1d(cout, cout, 5, padding=2), nn.ReLU(),
        nn.MaxPool1d(2),
    )


class AstroNet(nn.Module):
    def __init__(self, gw=(16, 32, 64, 128, 256), lw=(16, 32),
                 fc=512, p_drop=0.3):
        super().__init__()
        g, cin = [], 1
        for w in gw:
            g.append(conv_block(cin, w))
            cin = w
        self.gnet = nn.Sequential(*g)

        l, cin = [], 1
        for w in lw:
            l.append(conv_block(cin, w))
            cin = w
        self.lnet = nn.Sequential(*l)

        with torch.no_grad():
            ng = self.gnet(torch.zeros(1, 1, 2001)).numel()
            nl = self.lnet(torch.zeros(1, 1, 201)).numel()
        self.head = nn.Sequential(
            nn.Linear(ng + nl, fc), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(fc, fc), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(fc, 1),
        )
        self.nfeat = ng + nl

    def forward(self, g, l):
        g = self.gnet(g).flatten(1)
        l = self.lnet(l).flatten(1)
        return self.head(torch.cat([g, l], 1)).squeeze(1)


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def load(device):
    with h5py.File(H5, "r") as f:
        G = torch.from_numpy(f["global"][:]).float().unsqueeze(1)
        L = torch.from_numpy(f["local"][:]).float().unsqueeze(1)
        y = torch.from_numpy(f["label"][:].astype(np.float32))
        snr = f["snr"][:]
    s = np.load(SPLITS)
    idx = {k: torch.from_numpy(s[k]).long() for k in ["train", "val", "test"]}
    # ~60 MB total, so it all lives on the GPU and there is no loader to
    # bottleneck on.
    return G.to(device), L.to(device), y.to(device), snr, idx


def evaluate(model, G, L, y, idx, device, bs=512):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs].to(device)
            out.append(torch.sigmoid(model(G[b], L[b])).cpu())
    p = torch.cat(out).numpy()
    t = y[idx].cpu().numpy()
    return p, t


def metrics(p, t):
    return dict(pr_auc=float(average_precision_score(t, p)),
                roc_auc=float(roc_auc_score(t, p)))


# ----------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------

def run_seed(seed, G, L, y, idx, device, epochs, bs, lr, patience, quiet=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Without this, cuDNN picks algorithms non-deterministically and a
    # rerun at the same seed lands on slightly different numbers. Costs a
    # little speed; buys a result that can be reproduced exactly.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = AstroNet().to(device)
    tr, va = idx["train"], idx["val"]

    npos = float(y[tr].sum())
    nneg = float(len(tr) - npos)
    pos_weight = torch.tensor(nneg / npos, device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best = {"pr_auc": -1.0, "epoch": -1}
    best_state = None
    bad = 0
    g = torch.Generator().manual_seed(seed)
    last_loss = float("nan")

    for ep in range(epochs):
        model.train()
        perm = tr[torch.randperm(len(tr), generator=g)]
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs].to(device)
            gb, lb, yb = G[b], L[b], y[b]
            # label-preserving mirror, applied to both views together
            if torch.rand(1, generator=g).item() < 0.5:
                gb = torch.flip(gb, dims=[2])
                lb = torch.flip(lb, dims=[2])
            opt.zero_grad()
            loss = lossf(model(gb, lb), yb)
            loss.backward()
            opt.step()
            last_loss = loss.item()

        p, t = evaluate(model, G, L, y, va, device)
        m = metrics(p, t)
        if m["pr_auc"] > best["pr_auc"]:
            best = {**m, "epoch": ep}
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
        if not quiet and ep % 5 == 0:
            print(f"    ep {ep:3d}  loss {last_loss:.4f}  "
                  f"val PR-AUC {m['pr_auc']:.4f}  (best {best['pr_auc']:.4f})")

    model.load_state_dict(best_state)
    return model, best


def main(seeds, epochs, bs, lr, patience):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    G, L, y, snr, idx = load(device)
    print(f"[data] train {len(idx['train'])}  val {len(idx['val'])}  "
          f"test {len(idx['test'])}   device={device}")
    m0 = AstroNet()
    print(f"[model] {sum(p.numel() for p in m0.parameters()) / 1e6:.2f} M params, "
          f"{m0.nfeat} concat features")

    band = np.digitize(snr, [3.0, 7.0])
    results = []
    test_preds = {}
    for sd in seeds:
        t0 = time.time()
        print(f"\n[seed {sd}]")
        model, best = run_seed(sd, G, L, y, idx, device, epochs, bs, lr,
                               patience)
        # ---- test touched here, once, after selection is finished ----
        p, t = evaluate(model, G, L, y, idx["test"], device)
        m = metrics(p, t)
        tb = band[idx["test"].numpy()]
        strat = {}
        for bi, nm in [(0, "snr<3"), (1, "snr3-7"), (2, "snr>7")]:
            sel = tb == bi
            if sel.sum() > 10 and len(np.unique(t[sel])) > 1:
                strat[nm] = float(average_precision_score(t[sel], p[sel]))
        results.append(dict(seed=sd, val_pr_auc=best["pr_auc"],
                            best_epoch=best["epoch"], test=m, strat=strat,
                            secs=time.time() - t0))
        test_preds[f"seed{sd}"] = p
        print(f"  val PR-AUC {best['pr_auc']:.4f} (ep {best['epoch']})  ->  "
              f"TEST PR-AUC {m['pr_auc']:.4f}  ROC-AUC {m['roc_auc']:.4f}  "
              f"[{time.time() - t0:.0f}s]")
        print("  by SNR: " + "  ".join(f"{k} {v:.4f}" for k, v in strat.items()))

    pr = np.array([r["test"]["pr_auc"] for r in results])
    rc = np.array([r["test"]["roc_auc"] for r in results])
    print("\n" + "=" * 62)
    print(f"TEST PR-AUC  {pr.mean():.4f} +/- {pr.std():.4f}  "
          f"over {len(seeds)} seeds")
    print(f"TEST ROC-AUC {rc.mean():.4f} +/- {rc.std():.4f}")
    print("Stage 0 RF baseline PR-AUC 0.947")
    print("=" * 62)
    for bi in ["snr<3", "snr3-7", "snr>7"]:
        v = np.array([r["strat"][bi] for r in results if bi in r["strat"]])
        if len(v):
            print(f"  {bi:8s} PR-AUC {v.mean():.4f} +/- {v.std():.4f}")
    with open("cnn_results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    # Per-object test predictions, so the CNN-vs-RF disagreement analysis
    # (Stage 4 item 3) does not require retraining.
    np.savez("cnn_test_preds.npz", test_idx=idx["test"].numpy(),
             y_true=y[idx["test"]].cpu().numpy(), **test_preds)
    print("\n[done] wrote cnn_results.json, cnn_test_preds.npz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=12)
    a = ap.parse_args()
    main(a.seeds, a.epochs, a.bs, a.lr, a.patience)
