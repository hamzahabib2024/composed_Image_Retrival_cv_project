# =============================================================================
# COMPOSED IMAGE RETRIEVAL USING CLIP ON COCO 2017
# Computer Vision Final Lab Project
# =============================================================================
# Dataset path on Kaggle:
#   /kaggle/input/coco-dataset/coco_data/
#     ├── annotations/captions_train2017.json
#     ├── annotations/captions_val2017.json
#     ├── images/train2017/
#     └── images/val2017/
# =============================================================================

# ── CELL 1: Install dependencies ─────────────────────────────────────────────
import subprocess
subprocess.run(["pip", "install", "ftfy", "regex", "tqdm",
                "git+https://github.com/openai/CLIP.git"], check=True)

# ── CELL 2: Imports ───────────────────────────────────────────────────────────
import os
import sys
import json
import random
import math
import time
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

import clip

# ── CELL 3: Configuration ─────────────────────────────────────────────────────
class Config:
    # Kaggle dataset paths
    DATA_ROOT      = "/kaggle/input/coco-dataset/coco_data"
    TRAIN_IMG_DIR  = f"{DATA_ROOT}/images/train2017"
    VAL_IMG_DIR    = f"{DATA_ROOT}/images/val2017"
    TRAIN_ANN      = f"{DATA_ROOT}/annotations/captions_train2017.json"
    VAL_ANN        = f"{DATA_ROOT}/annotations/captions_val2017.json"

    # Model
    CLIP_MODEL     = "ViT-B/32"
    EMBED_DIM      = 512       # CLIP ViT-B/32 output dim
    PROJ_DIM       = 512       # projection head output dim
    HIDDEN_DIM     = 1024

    # Training
    BATCH_SIZE     = 64
    EPOCHS         = 15
    LR             = 3e-4
    WEIGHT_DECAY   = 1e-4
    ALPHA_INIT     = 0.5       # initial image/text blend weight
    TEMPERATURE    = 0.07      # InfoNCE temperature
    MAX_TRAIN_IMGS = 20000     # limit for speed (use None for full dataset)
    MAX_VAL_IMGS   = 5000      # limit for speed (use None for full dataset)
    SEED           = 42

    # Evaluation
    RECALL_AT_K    = [1, 5, 10]

    # Output
    SAVE_DIR       = "/kaggle/working"
    MODEL_CKPT     = f"{SAVE_DIR}/best_model.pt"

cfg = Config()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

random.seed(cfg.SEED)
np.random.seed(cfg.SEED)
torch.manual_seed(cfg.SEED)

# ── CELL 4: Load CLIP ─────────────────────────────────────────────────────────
clip_model, preprocess = clip.load(cfg.CLIP_MODEL, device=device)
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False
print(f"CLIP loaded: {cfg.CLIP_MODEL}")

# ── CELL 5: Dataset ───────────────────────────────────────────────────────────
class COCOCaptionDataset(Dataset):
    """
    Each item is (image_id, image_path, list_of_captions).
    We build a lookup: image_id → captions and image_id → path.
    """
    def __init__(self, img_dir, ann_file, transform, max_imgs=None):
        self.img_dir   = img_dir
        self.transform = transform

        with open(ann_file) as f:
            data = json.load(f)

        # id → filename
        self.id2file = {img["id"]: img["file_name"] for img in data["images"]}

        # id → captions list
        self.id2caps = {}
        for ann in data["annotations"]:
            iid = ann["image_id"]
            self.id2caps.setdefault(iid, []).append(ann["caption"])

        # keep only images that have a file on disk
        all_ids = [iid for iid in self.id2caps
                   if os.path.exists(os.path.join(img_dir, self.id2file[iid]))]
        if max_imgs:
            all_ids = all_ids[:max_imgs]

        self.image_ids = all_ids
        print(f"  Loaded {len(self.image_ids)} images from {ann_file}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        iid   = self.image_ids[idx]
        path  = os.path.join(self.img_dir, self.id2file[iid])
        img   = Image.open(path).convert("RGB")
        img   = self.transform(img)
        cap   = random.choice(self.id2caps[iid])  # random caption per call
        return img, cap, iid

print("Building datasets …")
train_ds = COCOCaptionDataset(cfg.TRAIN_IMG_DIR, cfg.TRAIN_ANN,
                               preprocess, cfg.MAX_TRAIN_IMGS)
val_ds   = COCOCaptionDataset(cfg.VAL_IMG_DIR,   cfg.VAL_ANN,
                               preprocess, cfg.MAX_VAL_IMGS)

train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                          shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE,
                          shuffle=False, num_workers=2, pin_memory=True)

# ── CELL 6: Composed Query Model ─────────────────────────────────────────────
class ComposedQueryModel(nn.Module):
    """
    Learns to compose a reference-image embedding + text-modification embedding
    into a single query vector used for retrieval.

    Architecture:
        [img_emb || txt_emb]  (1024-d concat)
            → LayerNorm
            → Linear(1024 → 1024) + GELU
            → Dropout
            → Linear(1024 → 512)
            → L2-normalise
    """
    def __init__(self, embed_dim=512, hidden_dim=1024, proj_dim=512):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(cfg.ALPHA_INIT))

        self.composer = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, img_emb, txt_emb):
        """
        img_emb, txt_emb : (B, 512) – already L2-normalised CLIP embeddings
        returns           : (B, 512) L2-normalised composed query
        """
        a = torch.sigmoid(self.alpha)
        # weighted sum as a warm-start signal
        fused = a * img_emb + (1 - a) * txt_emb
        # concat for the MLP
        cat   = torch.cat([img_emb, txt_emb], dim=-1)
        out   = self.composer(cat)
        # residual: add fused signal to MLP output
        out   = out + fused
        return F.normalize(out, dim=-1)


model = ComposedQueryModel(cfg.EMBED_DIM, cfg.HIDDEN_DIM, cfg.PROJ_DIM).to(device)
optimizer = Adam(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {total_params:,}")

# ── CELL 7: InfoNCE Loss ──────────────────────────────────────────────────────
def infonce_loss(query_emb, target_emb, temperature=cfg.TEMPERATURE):
    """
    Symmetric InfoNCE (contrastive) loss.
    query_emb  : (B, D)
    target_emb : (B, D)  – D-th target is the positive for D-th query
    """
    logits = (query_emb @ target_emb.T) / temperature   # (B, B)
    labels = torch.arange(logits.size(0), device=device)
    loss_q = F.cross_entropy(logits,   labels)           # query→target
    loss_t = F.cross_entropy(logits.T, labels)           # target→query
    return (loss_q + loss_t) / 2

# ── CELL 8: Training helpers ──────────────────────────────────────────────────
@torch.no_grad()
def encode_images(images):
    return F.normalize(clip_model.encode_image(images.to(device)).float(), dim=-1)

@torch.no_grad()
def encode_texts(captions):
    tokens = clip.tokenize(captions, truncate=True).to(device)
    return F.normalize(clip_model.encode_text(tokens).float(), dim=-1)


def train_one_epoch(epoch):
    """
    Synthetic composed-query training:
      • Within each batch we treat each image as a 'target'.
      • Its paired caption is the 'text modification'.
      • A random OTHER image in the batch is the 'reference image'.
      • The model must compose (ref_img + caption) to retrieve the target.
    """
    model.train()
    total_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS} [Train]",
                unit="batch", leave=True, dynamic_ncols=True)

    for step, (imgs, caps, _) in enumerate(pbar):
        B = imgs.size(0)
        if B < 2:
            continue

        img_emb = encode_images(imgs)       # (B,512)
        txt_emb = encode_texts(list(caps))  # (B,512)

        # reference images: roll by 1 so ref_img[i] ≠ target_img[i]
        ref_emb = torch.roll(img_emb, shifts=1, dims=0)

        # composed query: should be close to img_emb[i]
        query_emb = model(ref_emb, txt_emb)  # (B,512)

        loss = infonce_loss(query_emb, img_emb)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        avg_so_far = total_loss / (step + 1)
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "avg_loss": f"{avg_so_far:.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}"
        })
        sys.stdout.flush()

    pbar.close()
    return total_loss / len(train_loader)

# ── CELL 9: Gallery embedding ─────────────────────────────────────────────────
@torch.no_grad()
def build_gallery(dataset):
    """Pre-compute CLIP image embeddings for every image in the dataset."""
    model.eval()
    all_embs = []
    all_ids  = []
    loader   = DataLoader(dataset, batch_size=128, shuffle=False,
                          num_workers=2, pin_memory=True)
    pbar = tqdm(loader, desc="Building gallery embeddings",
                unit="batch", leave=True, dynamic_ncols=True)
    for imgs, _, iids in pbar:
        emb = encode_images(imgs)
        all_embs.append(emb.cpu())
        all_ids.extend(iids.tolist() if torch.is_tensor(iids) else iids)
        pbar.set_postfix({"embedded": len(all_ids)})
        sys.stdout.flush()
    pbar.close()
    gallery_embs = torch.cat(all_embs, dim=0)   # (N, 512)
    print(f"  Gallery ready: {gallery_embs.shape[0]} images, dim={gallery_embs.shape[1]}")
    sys.stdout.flush()
    return gallery_embs, all_ids

# ── CELL 10: Evaluation ───────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(dataset, gallery_embs, gallery_ids, k_list=[1, 5, 10]):
    """
    For each val image:
      - Use a random OTHER val image as the reference.
      - Use the target image's caption as the text modification.
      - Compose the query and retrieve top-K from the gallery.
      - Check if the true target is in top-K.
    """
    model.eval()
    id2idx = {iid: i for i, iid in enumerate(gallery_ids)}
    gallery_embs = gallery_embs.to(device)

    hits  = {k: 0 for k in k_list}
    total = 0

    loader = DataLoader(dataset, batch_size=64, shuffle=False,
                        num_workers=2, pin_memory=True)

    pbar = tqdm(loader, desc="Evaluating", unit="batch",
                leave=True, dynamic_ncols=True)
    for imgs, caps, iids in pbar:
        B = imgs.size(0)
        target_embs = encode_images(imgs)                # (B,512)
        txt_embs    = encode_texts(list(caps))           # (B,512)

        # reference: roll so ref ≠ target
        ref_embs = torch.roll(target_embs, shifts=1, dims=0)

        query_embs = model(ref_embs, txt_embs)           # (B,512)

        # similarity against the full gallery
        sims = query_embs @ gallery_embs.T               # (B, N)

        iids_list = iids.tolist() if torch.is_tensor(iids) else list(iids)
        for i, iid in enumerate(iids_list):
            if iid not in id2idx:
                continue
            true_idx = id2idx[iid]

            # mask out the reference image (rolled index)
            ref_iid = iids_list[(i - 1) % B]
            if ref_iid in id2idx:
                sims[i, id2idx[ref_iid]] = -1e9

            topk_max = max(k_list)
            _, top_indices = sims[i].topk(topk_max)
            top_indices = top_indices.tolist()

            for k in k_list:
                if true_idx in top_indices[:k]:
                    hits[k] += 1
            total += 1

        pbar.set_postfix({
            f"R@{k}": f"{hits[k]/max(total,1)*100:.1f}%" for k in k_list
        })
        sys.stdout.flush()

    pbar.close()
    recalls = {k: hits[k] / total * 100 for k in k_list}
    return recalls

# ── CELL 11: Training Loop ────────────────────────────────────────────────────
print("\n" + "="*60)
print("STARTING TRAINING")
print(f"  Epochs        : {cfg.EPOCHS}")
print(f"  Batch size    : {cfg.BATCH_SIZE}")
print(f"  Learning rate : {cfg.LR}")
print(f"  Device        : {device}")
print(f"  Train batches : {len(train_loader)}")
print(f"  Val batches   : {len(val_loader)}")
print("="*60)
sys.stdout.flush()

train_losses   = []
recall_history = {k: [] for k in cfg.RECALL_AT_K}
best_r10       = 0.0

print("\nStep 1 — Building validation gallery …")
sys.stdout.flush()
gallery_embs, gallery_ids = build_gallery(val_ds)
print(f"Gallery built: {len(gallery_ids)} images\n")
sys.stdout.flush()

epoch_bar = tqdm(range(cfg.EPOCHS), desc="Overall progress",
                 unit="epoch", position=0, leave=True, dynamic_ncols=True)

for epoch in epoch_bar:
    t0 = time.time()
    epoch_bar.set_description(f"Epoch {epoch+1}/{cfg.EPOCHS}")

    avg_loss = train_one_epoch(epoch)
    scheduler.step()
    train_losses.append(avg_loss)
    elapsed = time.time() - t0

    # evaluate every 2 epochs or on last epoch
    if (epoch + 1) % 2 == 0 or epoch == cfg.EPOCHS - 1:
        print(f"\n  → Evaluating after epoch {epoch+1} …")
        sys.stdout.flush()
        recalls = evaluate(val_ds, gallery_embs, gallery_ids, cfg.RECALL_AT_K)
        for k, v in recalls.items():
            recall_history[k].append((epoch + 1, v))

        r1  = recalls[1]
        r5  = recalls[5]
        r10 = recalls[10]
        print(f"  ┌─ Epoch {epoch+1:02d} Results ─────────────────────────")
        print(f"  │  Avg Train Loss : {avg_loss:.4f}")
        print(f"  │  Recall@1       : {r1:.2f}%")
        print(f"  │  Recall@5       : {r5:.2f}%")
        print(f"  │  Recall@10      : {r10:.2f}%  {'✓ TARGET MET' if r10>=40 else ''}")
        print(f"  │  Time/epoch     : {elapsed:.1f}s")
        print(f"  └──────────────────────────────────────────")
        sys.stdout.flush()

        epoch_bar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "R@10": f"{r10:.1f}%"
        })

        if r10 > best_r10:
            best_r10 = r10
            torch.save({
                "epoch"       : epoch + 1,
                "model_state" : model.state_dict(),
                "optimizer"   : optimizer.state_dict(),
                "recalls"     : recalls,
            }, cfg.MODEL_CKPT)
            print(f"  ✓ New best R@10 = {best_r10:.2f}% — checkpoint saved to {cfg.MODEL_CKPT}")
            sys.stdout.flush()
    else:
        print(f"  Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")
        sys.stdout.flush()
        epoch_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

epoch_bar.close()
print("\n" + "="*60)
print("TRAINING COMPLETE")
print(f"  Best Recall@10 : {best_r10:.2f}%")
print("="*60)
sys.stdout.flush()

# ── CELL 12: Plot Training Curves ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Training Results – Composed Image Retrieval (COCO)", fontsize=14)

# Loss curve
axes[0].plot(range(1, cfg.EPOCHS + 1), train_losses, 'b-o', markersize=4)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("InfoNCE Loss")
axes[0].set_title("Training Loss")
axes[0].grid(True, alpha=0.3)

# Recall curves
colors = {1: "green", 5: "orange", 10: "red"}
for k in cfg.RECALL_AT_K:
    if recall_history[k]:
        epochs_evaled, vals = zip(*recall_history[k])
        axes[1].plot(epochs_evaled, vals, f'-o', color=colors[k],
                     label=f"R@{k}", markersize=4)
axes[1].axhline(40, linestyle='--', color='gray', alpha=0.7, label="Target 40%")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Recall (%)")
axes[1].set_title("Recall@K on val2017")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{cfg.SAVE_DIR}/training_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved training_curves.png")

# ── CELL 13: Load best checkpoint ─────────────────────────────────────────────
ckpt = torch.load(cfg.MODEL_CKPT, map_location=device)
model.load_state_dict(ckpt["model_state"])
print(f"Loaded best checkpoint from epoch {ckpt['epoch']}")
best_recalls = {f"R@{k}": f"{v:.2f}%" for k, v in ckpt["recalls"].items()}
print(f"Best recalls: {best_recalls}")

# ── CELL 14: Qualitative Demo ─────────────────────────────────────────────────
@torch.no_grad()
def retrieve_top_k(ref_img_tensor, text_query, gallery_embs, gallery_ids,
                   gallery_dataset, k=10):
    """Given a reference image tensor and a text string, return top-k image paths."""
    model.eval()
    ref_emb  = encode_images(ref_img_tensor.unsqueeze(0))        # (1,512)
    txt_emb  = encode_texts([text_query])                         # (1,512)
    query    = model(ref_emb, txt_emb)                            # (1,512)

    gallery_on_device = gallery_embs.to(device)
    sims = (query @ gallery_on_device.T).squeeze(0)               # (N,)
    _, top_idx = sims.topk(k)

    id2path = {iid: os.path.join(gallery_dataset.img_dir,
                                  gallery_dataset.id2file[iid])
               for iid in gallery_ids}
    results = [(gallery_ids[i], id2path[gallery_ids[i]], sims[i].item())
               for i in top_idx.tolist()]
    return results


def demo_retrieval(sample_idx=0, text_query=None):
    """Visualise: reference image | query text | top-5 retrieved images."""
    img_tensor, default_cap, iid = val_ds[sample_idx]
    text_query = text_query or default_cap

    results = retrieve_top_k(img_tensor, text_query,
                              gallery_embs, gallery_ids, val_ds, k=5)

    fig = plt.figure(figsize=(18, 4))
    gs  = gridspec.GridSpec(1, 7, figure=fig)

    # Reference image
    ref_path = os.path.join(val_ds.img_dir, val_ds.id2file[iid])
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(Image.open(ref_path).convert("RGB"))
    ax0.set_title("Reference\nImage", fontsize=9)
    ax0.axis("off")

    # Query text
    ax1 = fig.add_subplot(gs[1])
    ax1.text(0.5, 0.5, f'Text Query:\n"{text_query}"',
             ha='center', va='center', wrap=True, fontsize=7,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    ax1.axis("off")

    # Top-5 retrieved
    for j, (ret_id, ret_path, score) in enumerate(results):
        ax = fig.add_subplot(gs[j + 2])
        ax.imshow(Image.open(ret_path).convert("RGB"))
        ax.set_title(f"#{j+1}\nsim={score:.3f}", fontsize=8)
        ax.axis("off")

    fig.suptitle("Composed Image Retrieval Demo", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{cfg.SAVE_DIR}/demo_retrieval_{sample_idx}.png",
                dpi=150, bbox_inches="tight")
    plt.show()


# Run demos on a few validation samples
for demo_idx in [0, 10, 50, 100, 200]:
    print(f"Demo for val sample index {demo_idx}")
    demo_retrieval(demo_idx)

# ── CELL 15: Final Recall Summary ─────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL EVALUATION SUMMARY")
print("="*60)
final_recalls = evaluate(val_ds, gallery_embs, gallery_ids, cfg.RECALL_AT_K)
for k, v in final_recalls.items():
    status = "✓" if v >= 40 else "✗"
    print(f"  {status} Recall@{k:2d}: {v:.2f}%")
print(f"\nTarget Recall@10 ≥ 40%: {'ACHIEVED ✓' if final_recalls[10] >= 40 else 'Not yet – consider more epochs'}")
print("="*60)
