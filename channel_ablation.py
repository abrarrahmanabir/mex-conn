import os
import argparse
from PIL import Image
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import MultiHeadUNet
from matplotlib_venn import venn3
from train import MultiOrganelleSegDataset
from sklearn.metrics import jaccard_score, f1_score, recall_score
from sklearn.metrics import mutual_info_score


# For organelles, keep the one to train and comment out the rest two

ORGANELLES = ("fusiform-vesicles", "mitochondria", "lysosomes")    # for urocell
ORGANELLES = ("membranes", "mitochondria", "synapses")             # for drosophila
ORGANELLES =  ("membranes", "mitochondria", "vesicles")            # for multiclass epfl
DOMAIN = None

def dice_coef(pred, target, smooth=1.):
    pred = (pred > 0.5).float()
    intersect = (pred * target).sum()
    return (2. * intersect + smooth) / (pred.sum() + target.sum() + smooth)

def variation_of_information(segA, segB, eps=1e-8):
    a = segA.flatten()
    b = segB.flatten()
    H_a = -np.sum([np.mean(a == v) * np.log2(np.mean(a == v) + eps) for v in [0, 1]])
    H_b = -np.sum([np.mean(b == v) * np.log2(np.mean(b == v) + eps) for v in [0, 1]])
    I = mutual_info_score(a, b)
    VI = H_a + H_b - 2 * I
    return VI


def find_topk_sets(model, loader, device, K=30, org_names=('membranes','mitochondria','synapses')):
    """
    Mechanistic interpretability: Identify which encoder channels are most important
    for each organelle-specific decoder head, using gradient-based channel importance.

    Args:
        model: Multi-head U-Net model (with shared encoder and multiple decoder heads).
        loader: DataLoader yielding (image, mask) pairs.
        device: Device to run computation on (CPU or CUDA).
        K: Number of top channels to consider per head.
        org_names: List of organelle names (matching decoder heads).

    Returns:
        topk_per_head: List of sets, top-K channel indices for each head.
        shared: Set of channel indices shared (important) across all heads.
        head_specific: List of sets, channels unique to each head's top-K set.
        C: Total number of encoder channels at the deepest layer.
    """
    encoder = model.encoder
    # Store importance per channel per head (list of [batch x channels])
    all_channel_importances = [ [] for _ in org_names ]
    print("Collecting gradients for all channels...")

    # Loop over all batches
    for idx, (img_patch, mask_patch) in enumerate(tqdm(loader)):
        img_patch = img_patch.to(device).requires_grad_(True)
        feats = encoder(img_patch)          # Forward pass: get encoder feature maps (multi-level)
        deep_feat = feats[-1]               # Use deepest feature map (shape: B x C x H x W)
        outs = []
        for dec in model.decoders:
            outs.append(dec(feats))         # Get logits from each decoder head

        out = torch.cat(outs, dim=1)        # Concatenate outputs: (B x num_heads x H x W)

        # Mechanistic interpretability: For each head, compute gradient
        for head_idx in range(len(org_names)):
            model.zero_grad()
            deep_feat.retain_grad()
            # Take the scalar sum of output logits for this head across batch & spatial dims
            head_out = out[:, head_idx].sum()
            # Backprop: Get gradient of output (scalar) w.r.t. encoder channels
            head_out.backward(retain_graph=True)
            grad = deep_feat.grad.detach().abs().cpu().numpy()  # Shape: (B, C, H, W)
            # Mean absolute gradient across batch and spatial dims: gives channel importance
            grad = grad.mean(axis=(0,2,3))                     # Shape: (C,)
            all_channel_importances[head_idx].append(grad)
            deep_feat.grad.zero_()

    # Compute average importance per channel for each head
    avg_importances = [ np.stack(x).mean(axis=0) for x in all_channel_importances ]  # List: [C] x num_heads

    # Top-K channel indices for each head
    topk_per_head = [ set(np.argsort(imp)[-K:]) for imp in avg_importances ]

    # Shared channels: channels present in top-K of ALL heads (intersection)
    shared = set.intersection(*topk_per_head)

    # Head-specific channels: those present only in top-K of one head but not the others
    head_specific = []
    for i in range(len(org_names)):
        only_this = topk_per_head[i] - set.union(*[topk_per_head[j] for j in range(len(org_names)) if j != i])
        head_specific.append(sorted(list(only_this)))

    # Visualize overlap as a heatmap (rows=heads, cols=channels)
    C = len(avg_importances[0])
    overlap_matrix = np.zeros((len(org_names), C), dtype=bool)
    for i, tk in enumerate(topk_per_head):
        overlap_matrix[i, list(tk)] = True
    plt.figure(figsize=(14, 3))
    plt.imshow(overlap_matrix, aspect='auto', cmap='viridis')
    plt.yticks(range(len(org_names)), org_names)
    plt.xlabel("Encoder Channel")
    plt.title(f"Top-{K} Encoder Channels Used by Each Head")
    plt.colorbar(label='Top-K Channel (1=True)')
    plt.tight_layout()
    plt.savefig(f'./plots/{DOMAIN}_heatmap.png', bbox_inches='tight', dpi=600)

    # Venn diagram for overlap (if 3 heads)
    if len(org_names) == 3:
        from matplotlib_venn import venn3
        sets = [set(x) for x in topk_per_head]
        plt.figure(figsize=(6,5))
        venn3(sets, set_labels=org_names)
        plt.title(f"Venn Diagram: Top-{K} Encoder Channel Overlap")
        plt.savefig(f'./plots/{DOMAIN}_venn.png', bbox_inches='tight', dpi=600)
        print(f"\nShared by all 3: {len(shared)}")
        print("Shared encoder channels:", sorted(shared))

    return topk_per_head, shared, head_specific, C

# -----------------------------------------------------------------------------

@torch.no_grad()
def evaluate_with_channel_ablation(model, loader, device, ablate_channels):
    """
    Evaluates the segmentation model after ablating (zeroing out) specific encoder channels.

    Args:
        model: Trained MultiHeadUNet model.
        loader: DataLoader over test images/masks.
        device: CPU or CUDA.
        ablate_channels: Iterable of channel indices to set to zero (simulate removal).

    Returns:
        results: Dict of segmentation metrics (dice, iou, f1, recall, vi) per organelle and averaged.
    """
    model.eval()
    dices = [[] for _ in range(3)]
    ious = [[] for _ in range(3)]
    f1s = [[] for _ in range(3)]
    recalls = [[] for _ in range(3)]
    vis = [[] for _ in range(3)]
    encoder = model.encoder

    # Iterate through test set
    for imgs, masks in tqdm(loader, desc="Eval normal vs ablated", leave=False):
        imgs = imgs.to(device)
        masks = masks.to(device)
        feats = encoder(imgs)
        feats_ablate = [f.clone() for f in feats]


        # Mechanistic ablation: zero out chosen encoder channels at deepest layer

        if ablate_channels and len(ablate_channels) > 0:
            idxs = torch.tensor(list(ablate_channels), device=feats_ablate[-1].device)
            feats_ablate[-1][:, idxs, :, :] = 0.0


        outs_ablate = []


        for dec in model.decoders:
            outs_ablate.append(dec(feats_ablate))
        preds_ablate = torch.sigmoid(torch.cat(outs_ablate, dim=1))
        preds_bin = (preds_ablate > 0.5).float()
        for c in range(3):
            for i in range(imgs.shape[0]):
                p_np = preds_bin[i, c].cpu().squeeze().numpy()
                m_np = masks[i, c].cpu().squeeze().numpy()
                dices[c].append(dice_coef(torch.tensor(p_np), torch.tensor(m_np)).item())
                ious[c].append(jaccard_score(m_np.flatten(), p_np.flatten(), zero_division=1))
                f1s[c].append(f1_score(m_np.flatten(), p_np.flatten(), zero_division=1))
                recalls[c].append(recall_score(m_np.flatten(), p_np.flatten(), zero_division=1))
                vis[c].append(variation_of_information(m_np, p_np))


    results = {}


    for c, name in enumerate(ORGANELLES):
        results[f"{name}_dice"] = np.mean(dices[c])
        results[f"{name}_iou"] = np.mean(ious[c])
        results[f"{name}_f1"] = np.mean(f1s[c])
        results[f"{name}_recall"] = np.mean(recalls[c])
        results[f"{name}_vi"] = np.mean(vis[c])


    #  return average metrics

    results["avg_dice"] = np.mean([np.mean(d) for d in dices])
    results["avg_iou"] = np.mean([np.mean(i) for i in ious])
    results["avg_f1"] = np.mean([np.mean(f) for f in f1s])
    results["avg_recall"] = np.mean([np.mean(r) for r in recalls])
    results["avg_vi"] = np.mean([np.mean(v) for v in vis])
    return results



@torch.no_grad()
def evaluate_normal(model, loader, device):
    model.eval()
    dices = [[] for _ in range(3)]
    ious = [[] for _ in range(3)]
    f1s = [[] for _ in range(3)]
    recalls = [[] for _ in range(3)]
    vis = [[] for _ in range(3)]
    for imgs, masks in tqdm(loader, desc="Eval NORMAL", leave=False):
        imgs = imgs.to(device)
        masks = masks.to(device)
        preds = model(imgs)
        preds = torch.sigmoid(preds)
        preds_bin = (preds > 0.5).float()
        for c in range(3):
            for i in range(imgs.shape[0]):
                p_np = preds_bin[i, c].cpu().squeeze().numpy()
                m_np = masks[i, c].cpu().squeeze().numpy()
                dices[c].append(dice_coef(torch.tensor(p_np), torch.tensor(m_np)).item())
                ious[c].append(jaccard_score(m_np.flatten(), p_np.flatten(), zero_division=1))
                f1s[c].append(f1_score(m_np.flatten(), p_np.flatten(), zero_division=1))
                recalls[c].append(recall_score(m_np.flatten(), p_np.flatten(), zero_division=1))
                vis[c].append(variation_of_information(m_np, p_np))
    results = {}
    for c, name in (ORGANELLES):
        results[f"{name}_dice"] = np.mean(dices[c])
        results[f"{name}_iou"] = np.mean(ious[c])
        results[f"{name}_f1"] = np.mean(f1s[c])
        results[f"{name}_recall"] = np.mean(recalls[c])
        results[f"{name}_vi"] = np.mean(vis[c])
    results["avg_dice"] = np.mean([np.mean(d) for d in dices])
    results["avg_iou"] = np.mean([np.mean(i) for i in ious])
    results["avg_f1"] = np.mean([np.mean(f) for f in f1s])
    results["avg_recall"] = np.mean([np.mean(r) for r in recalls])
    results["avg_vi"] = np.mean([np.mean(v) for v in vis])
    return results

def print_metrics(label, results):
    print(f"\n------ {label} ------")
    for k in (ORGANELLES):
        print(f"{k:15} | Dice: {results[f'{k}_dice']:.4f}  IoU: {results[f'{k}_iou']:.4f}  F1: {results[f'{k}_f1']:.4f}  Recall: {results[f'{k}_recall']:.4f}  VI: {results[f'{k}_vi']:.4f}")
    print("---------------------------------------------------------")
    print(f"Avg Dice:   {results['avg_dice']:.4f}   Avg IoU:  {results['avg_iou']:.4f}   Avg F1: {results['avg_f1']:.4f}   Avg Recall: {results['avg_recall']:.4f}   Avg VI: {results['avg_vi']:.4f}")



import pandas as pd

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--domain', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--top_k', type=int, default=30)
    args = parser.parse_args()
    DOMAIN = args.domain
    domain_path = os.path.join(args.data_root, args.domain)


    organelles = ORGANELLES
    test_ds = MultiOrganelleSegDataset(domain_path, "test", organelles, patch_size=256, stride=128)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    model = MultiHeadUNet(in_ch=1, base_features=[32, 64, 128, 256], out_ch_per_head=1, num_heads=3).to(args.device)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))

    # ---- 1. Channel analysis & plotting ----
    topk_per_head, shared, head_specific, total_channels = find_topk_sets(model, test_loader, args.device, K=args.top_k, org_names=organelles)

    print("\n---- Running Ablation Experiments ----")

    # -- Exp 1: Drop shared channels (intersection: all 3 heads) --
    print(f"\n[Experiment 1] Drop SHARED channels ({len(shared)} channels):")
    ablate_shared_results = evaluate_with_channel_ablation(model, test_loader, args.device, shared)
    print_metrics("Ablate SHARED", ablate_shared_results)

    # -- Exp 2: Drop head-specific channels (present in only one head's topK) --
    head_specific_flat = set()
    for lst in head_specific:
        head_specific_flat |= set(lst)
    print(f"\n[Experiment 2] Drop HEAD-SPECIFIC channels (present in only one topK set): {len(head_specific_flat)} channels")
    ablate_headspec_results = evaluate_with_channel_ablation(model, test_loader, args.device, head_specific_flat)
    print_metrics("Ablate HEAD-SPECIFIC", ablate_headspec_results)

    metric_keys = ["dice", "iou", "f1", "recall", "vi"]
    metric_pretty = {"dice": "Dice", "iou": "IoU", "f1": "F1", "recall": "Recall", "vi": "VI"}
    exp_names = ["Ablate Shared", "Ablate Head-Specific"]
    results_all = [ablate_shared_results, ablate_headspec_results]

    rows = []
    for exp, res in zip(exp_names, results_all):
        for o in organelles:
            row = {"Experiment": exp, "Organelle": o.capitalize()}
            for m in metric_keys:
                row[metric_pretty[m]] = res[f"{o}_{m}"]
            rows.append(row)
        # Avg row
        row = {"Experiment": exp, "Organelle": "AVG"}
        for m in metric_keys:
            row[metric_pretty[m]] = res[f"avg_{m}"]
        rows.append(row)

    out_csv = f"{args.domain}_shared_head.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\nResults saved to {out_csv}")

if __name__ == '__main__':
    main()






'''

python channel_ablation.py --data_root data --domain drosophila-vnc --model_path ./models/drosophila/model.pth --batch_size 8 --top_k 100 --device cuda

python channel_ablation.py --data_root data --domain multiclass --model_path ./models/multiclass/model.pth --batch_size 8 --top_k 100 --device cuda

python channel_ablation.py --data_root data --domain urocell_3 --model_path ./models/urocell_3/model.pth --batch_size 8 --top_k 100 --device cuda


'''
