# =============================================================================
# CONSOLE INTERFACE — Composed Image Retrieval using CLIP
# Run this AFTER training is complete (best_model.pt must exist)
# Usage: python app.py
# =============================================================================

import os
import sys
import json
import random
import subprocess

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# ── Install OpenCV / Transformers if needed ──────────────────────────────────
try:
    import cv2
except ImportError:
    subprocess.run(["pip", "install", "opencv-python"], check=True)
    import cv2

try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    subprocess.run(["pip", "install", "transformers"], check=True)
    from transformers import CLIPProcessor, CLIPModel

# ── Install Gradio if needed ──────────────────────────────────────────────────
try:
    import gradio as gr
except ImportError:
    subprocess.run(["pip", "install", "gradio"], check=True)
    import gradio as gr

# ── Paths (Kaggle) ────────────────────────────────────────────────────────────
DATA_ROOT    = "/kaggle/input/coco-dataset/coco_data"
VAL_IMG_DIR  = f"/kaggle/input/datasets/salargamer/coco-dataset/coco_data/images/val2017"
VAL_ANN      = f"/kaggle/input/datasets/salargamer/coco-dataset/coco_data/annotations/captions_val2017.json"
MODEL_CKPT   = "/kaggle/working/best_model.pt"
SAVE_DIR     = "/kaggle/working"
EMBED_DIM    = 512
HIDDEN_DIM   = 1024
device       = "cuda" if torch.cuda.is_available() else "cpu"


# ── Model definition (must match training) ────────────────────────────────────
class ComposedQueryModel(nn.Module):
    def __init__(self, embed_dim=512, hidden_dim=1024, proj_dim=512):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.composer = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, img_emb, txt_emb):
        a     = torch.sigmoid(self.alpha)
        fused = a * img_emb + (1 - a) * txt_emb
        cat   = torch.cat([img_emb, txt_emb], dim=-1)
        out   = self.composer(cat)
        return F.normalize(out + fused, dim=-1)

# ── Load CLIP + model ─────────────────────────────────────────────────────────
def load_models():
    print("Loading CLIP (ViT-B/32) …")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    print(f"Loading checkpoint: {MODEL_CKPT}")
    ckpt  = torch.load(MODEL_CKPT, map_location=device)
    model = ComposedQueryModel().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ep      = ckpt.get("epoch", "?")
    recalls = ckpt.get("recalls", {})
    print(f"  Checkpoint from epoch {ep}")
    for k, v in recalls.items():
        print(f"  Recall@{k}: {v:.2f}%")
    return clip_model, processor, model

# ── Build gallery ─────────────────────────────────────────────────────────────
def build_gallery(clip_model, preprocess):
    print("\nBuilding image gallery from val2017 …")
    with open(VAL_ANN) as f:
        data = json.load(f)

    id2file = {img["id"]: img["file_name"] for img in data["images"]}
    valid_ids = [iid for iid, fn in id2file.items()
                 if os.path.exists(os.path.join(VAL_IMG_DIR, fn))]

    all_embs = []
    all_ids  = []
    batch_size = 128

    pbar = tqdm(range(0, len(valid_ids), batch_size),
                desc="Embedding gallery", unit="batch")
    for i in pbar:
        batch_ids = valid_ids[i:i + batch_size]
        imgs = []
        for iid in batch_ids:
            path = os.path.join(VAL_IMG_DIR, id2file[iid])
            img = cv2.imread(path)
            if img is None:
                continue
            imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if len(imgs) == 0:
            continue
        inputs = preprocess(images=imgs, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(device)
        with torch.no_grad():
            emb = F.normalize(clip_model.get_image_features(pixel_values=pixel_values).float(), dim=-1)
        all_embs.append(emb.cpu())
        all_ids.extend(batch_ids[: len(imgs)])
        pbar.set_postfix({"done": len(all_ids)})

    gallery_embs = torch.cat(all_embs, dim=0)
    print(f"  Gallery ready: {len(all_ids)} images\n")
    return gallery_embs, all_ids, id2file

# ── Retrieval ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def retrieve(ref_input, text_query, clip_model, preprocess, model,
             gallery_embs, gallery_ids, id2file, top_k=5):
    # encode reference image
    ref_emb, _ = encode_reference(ref_input, preprocess, clip_model)

    # encode text
    text_inputs = preprocess(text=[text_query], return_tensors="pt", padding=True, truncation=True)
    input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)
    txt_emb = F.normalize(
        clip_model.get_text_features(input_ids=input_ids,
                                     attention_mask=attention_mask).float(),
        dim=-1)

    # compose
    query = model(ref_emb, txt_emb)

    # similarity search
    sims = (query @ gallery_embs.to(device).T).squeeze(0)
    _, top_idx = sims.topk(top_k)

    results = []
    for i in top_idx.tolist():
        iid  = gallery_ids[i]
        path = os.path.join(VAL_IMG_DIR, id2file[iid])
        results.append((iid, path, sims[i].item()))
    return results

# ── Reference image encoding helper ──────────────────────────────────────────
def encode_reference(ref_input, preprocess, clip_model):
    if isinstance(ref_input, str):
        img = cv2.imread(ref_input)
        if img is None:
            raise FileNotFoundError(f"Unable to load image: {ref_input}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(ref_input, np.ndarray):
        img = ref_input.astype("uint8")
    else:
        raise ValueError("Unsupported reference image type")

    inputs = preprocess(images=img, return_tensors="pt")
    ref_tensor = inputs.pixel_values.to(device)
    with torch.no_grad():
        ref_emb = F.normalize(clip_model.get_image_features(pixel_values=ref_tensor).float(), dim=-1)
    return ref_emb, img

# ── Display results ───────────────────────────────────────────────────────────
def display_results(ref_path, text_query, results, save_path=None):
    n = len(results)
    fig = plt.figure(figsize=(4 * (n + 2), 4))
    gs  = gridspec.GridSpec(1, n + 2, figure=fig,
                            wspace=0.3, hspace=0.1)

    # Reference
    ref_img = cv2.imread(ref_path)
    if ref_img is None:
        raise FileNotFoundError(f"Unable to load image: {ref_path}")
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(ref_img)
    ax0.set_title("Reference\nImage", fontsize=10, fontweight="bold")
    ax0.axis("off")

    # Query text
    ax1 = fig.add_subplot(gs[1])
    ax1.text(0.5, 0.5, f'Text Query:\n"{text_query}"',
             ha="center", va="center", wrap=True, fontsize=8,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF3CD",
                       edgecolor="#FFC107", linewidth=1.5))
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    ax1.axis("off")

    # Retrieved
    for j, (iid, path, score) in enumerate(results):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Unable to load image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax = fig.add_subplot(gs[j + 2])
        ax.imshow(img)
        ax.set_title(f"#{j+1}  sim={score:.3f}", fontsize=8)
        ax.axis("off")

    fig.suptitle("Composed Image Retrieval Result",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()

# ── Gallery image picker (random from val) ────────────────────────────────────
def pick_random_gallery_image(gallery_ids, id2file):
    iid  = random.choice(gallery_ids)
    path = os.path.join(VAL_IMG_DIR, id2file[iid])
    return path, iid

# ── Console Menu ──────────────────────────────────────────────────────────────
def print_banner():
    print("\n" + "═"*60)
    print("  COMPOSED IMAGE RETRIEVAL SYSTEM")
    print("  CLIP ViT-B/32  |  COCO val2017 Gallery")
    print("═"*60)

def menu():
    print_banner()
    print("\n[1] Query with a random gallery image as reference")
    print("[2] Query with a custom image path as reference")
    print("[3] Run evaluation demo (5 random samples)")
    print("[4] Show gallery statistics")
    print("[0] Exit\n")
    return input("Select option: ").strip()

def run_app():
    clip_model, preprocess, model = load_models()
    gallery_embs, gallery_ids, id2file = build_gallery(clip_model, preprocess)

    query_count = 0

    while True:
        choice = menu()

        # ── Option 1: Random reference ────────────────────────────────────
        if choice == "1":
            ref_path, ref_iid = pick_random_gallery_image(gallery_ids, id2file)
            print(f"\nReference image: {os.path.basename(ref_path)}")

            # show reference
            img = cv2.imread(ref_path)
            if img is None:
                print(f"Unable to load image: {ref_path}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(4, 4))
            plt.imshow(img)
            plt.title("Your Reference Image")
            plt.axis("off")
            plt.tight_layout()
            plt.show()

            text_query = input("Enter text modification query: ").strip()
            if not text_query:
                print("Query cannot be empty.")
                continue

            top_k = input("Top-K results to show [default 5]: ").strip()
            top_k = int(top_k) if top_k.isdigit() else 5

            print("\nSearching …")
            results = retrieve(ref_path, text_query, clip_model, preprocess,
                               model, gallery_embs, gallery_ids, id2file, top_k)

            query_count += 1
            save_path = f"{SAVE_DIR}/query_{query_count:03d}.png"
            display_results(ref_path, text_query, results, save_path)

            print("\nTop results:")
            for rank, (iid, path, score) in enumerate(results, 1):
                print(f"  #{rank}  sim={score:.4f}  {os.path.basename(path)}")

        # ── Option 2: Custom reference path ──────────────────────────────
        elif choice == "2":
            ref_path = input("Enter full path to reference image: ").strip()
            if not os.path.exists(ref_path):
                print(f"File not found: {ref_path}")
                continue

            text_query = input("Enter text modification query: ").strip()
            if not text_query:
                print("Query cannot be empty.")
                continue

            top_k = input("Top-K results to show [default 5]: ").strip()
            top_k = int(top_k) if top_k.isdigit() else 5

            print("\nSearching …")
            results = retrieve(ref_path, text_query, clip_model, preprocess,
                               model, gallery_embs, gallery_ids, id2file, top_k)

            query_count += 1
            save_path = f"{SAVE_DIR}/query_{query_count:03d}.png"
            display_results(ref_path, text_query, results, save_path)

            print("\nTop results:")
            for rank, (iid, path, score) in enumerate(results, 1):
                print(f"  #{rank}  sim={score:.4f}  {os.path.basename(path)}")

        # ── Option 3: Demo — 5 random samples ────────────────────────────
        elif choice == "3":
            print("\nRunning demo on 5 random gallery images …")
            sample_queries = [
                "a dog running in a park",
                "a person eating at a restaurant",
                "a sports car on a highway",
                "children playing outdoors",
                "a boat on the water",
            ]
            for i, query in enumerate(sample_queries):
                ref_path, _ = pick_random_gallery_image(gallery_ids, id2file)
                results = retrieve(ref_path, query, clip_model, preprocess,
                                   model, gallery_embs, gallery_ids, id2file, top_k=5)
                save_path = f"{SAVE_DIR}/demo_{i+1:02d}.png"
                print(f"\n  Demo {i+1}/5 — Query: \"{query}\"")
                display_results(ref_path, query, results, save_path)

        # ── Option 4: Gallery stats ───────────────────────────────────────
        elif choice == "4":
            print("\n── Gallery Statistics ──────────────────────────")
            print(f"  Total images : {len(gallery_ids)}")
            print(f"  Embedding dim: {gallery_embs.shape[1]}")
            print(f"  Device       : {device}")
            print(f"  Model ckpt   : {MODEL_CKPT}")
            norms = gallery_embs.norm(dim=1)
            print(f"  Emb norm  — min: {norms.min():.4f} "
                  f"max: {norms.max():.4f} mean: {norms.mean():.4f}")
            print("─"*48)

        # ── Exit ──────────────────────────────────────────────────────────
        elif choice == "0":
            print("\nExiting. Goodbye!\n")
            break
        else:
            print("Invalid option. Please choose 0–4.")

# ── Gradio interface ──────────────────────────────────────────────────────────
def launch_gradio(clip_model, preprocess, model, gallery_embs, gallery_ids,
                  id2file):
    def gradio_search(ref_image, text_query, top_k):
        if ref_image is None:
            return None, "Please upload a reference image."
        if not text_query or not text_query.strip():
            return None, "Please enter a text query."

        try:
            results = retrieve(ref_image, text_query, clip_model, preprocess,
                               model, gallery_embs, gallery_ids, id2file,
                               top_k=int(top_k))
        except Exception as exc:
            return None, f"Error during retrieval: {exc}"

        gallery = []
        for _, path, _ in results:
            img = cv2.imread(path)
            if img is None:
                continue
            gallery.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        captions = [f"#{i+1}: {os.path.basename(path)} (sim={score:.3f})"
                    for i, (_, path, score) in enumerate(results)]
        return gallery, "\n".join(captions)

    with gr.Blocks(title="Composed Image Retrieval") as demo:
        gr.Markdown("# Composed Image Retrieval with CLIP and COCO")
        gr.Markdown(
            "Upload a reference image and enter a text modification query to retrieve similar COCO validation images.")

        with gr.Row():
            ref_input = gr.Image(type="numpy", label="Reference Image")
            query_input = gr.Textbox(lines=2, label="Text Query",
                                     placeholder="e.g. a dog running in a park")
        with gr.Row():
            top_k_input = gr.Slider(minimum=1, maximum=10, value=5,
                                    step=1, label="Top-K results")
        run_button = gr.Button("Search")
        result_gallery = gr.Gallery(label="Retrieved Images").style(grid=[5])
        result_text = gr.Textbox(label="Result Metadata", interactive=False)

        run_button.click(fn=gradio_search,
                         inputs=[ref_input, query_input, top_k_input],
                         outputs=[result_gallery, result_text])

    demo.launch(server_name="0.0.0.0")

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--gradio", "--ui", "--web"):
        clip_model, preprocess, model = load_models()
        gallery_embs, gallery_ids, id2file = build_gallery(clip_model, preprocess)
        launch_gradio(clip_model, preprocess, model, gallery_embs, gallery_ids, id2file)
    else:
        run_app()
