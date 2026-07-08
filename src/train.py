"""
train.py
Training pipeline for the Drug Interaction GNN.

What happens here:
    1. Load dataset
    2. Split into train / validation sets
    3. Run training loop (forward → loss → backward → update weights)
    4. Evaluate on validation set each epoch
    5. Save the best model
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch.utils.data import random_split
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import DrugInteractionDataset, LABEL_NAMES
from model   import DrugInteractionGNN

# ── Config ─────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH    = os.path.join(BASE_DIR, "data", "processed", "real_interactions.csv")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
MODEL_PATH  = os.path.join(MODEL_DIR, "best_model.pt")
os.makedirs(MODEL_DIR, exist_ok=True)

# Training hyperparameters
EPOCHS      = 200
LR          = 0.001       # learning rate
HIDDEN_DIM  = 128
NUM_CLASSES = 4
TRAIN_SPLIT = 0.8         # 80% train, 20% validation
SEED        = 42


def collate_fn(batch):
    """
    Custom collate — converts a list of sample dicts into batched PyG objects.
    This is needed because PyG graphs can't be stacked like normal tensors.
    """
    graphs_a = Batch.from_data_list([s["graph_a"] for s in batch])
    graphs_b = Batch.from_data_list([s["graph_b"] for s in batch])
    labels   = torch.tensor([s["label"] for s in batch], dtype=torch.long)
    return graphs_a, graphs_b, labels


def collate_single(samples):
    """Collate for DataLoader — wraps collate_fn."""
    return collate_fn(samples)


def train_epoch(model, data, optimizer, device):
    """Run one full training epoch. Returns average loss."""
    model.train()
    total_loss = 0.0

    # With a tiny dataset we train on all samples in one batch
    graphs_a, graphs_b, labels = collate_fn(data)
    graphs_a = graphs_a.to(device)
    graphs_b = graphs_b.to(device)
    labels   = labels.to(device)

    optimizer.zero_grad()
    logits = model(graphs_a, graphs_b)
    loss   = F.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model, data, device):
    """Evaluate on a dataset split. Returns loss, accuracy, predictions."""
    model.eval()

    graphs_a, graphs_b, labels = collate_fn(data)
    graphs_a = graphs_a.to(device)
    graphs_b = graphs_b.to(device)
    labels   = labels.to(device)

    logits = model(graphs_a, graphs_b)
    loss   = F.cross_entropy(logits, labels).item()

    preds = torch.argmax(logits, dim=1)
    acc   = (preds == labels).float().mean().item()

    return loss, acc, preds.cpu().tolist(), labels.cpu().tolist()


def main():
    print("=" * 55)
    print("  Drug Interaction GNN — Training")
    print("=" * 55)

    # ── Device ─────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")

    # ── Dataset ────────────────────────────────────────────────────────────
    print("\n  Loading dataset …")
    dataset = DrugInteractionDataset(CSV_PATH, verbose=True)

    if len(dataset) < 4:
        print("  ❌ Dataset too small to split. Add more drug pairs.")
        return

    # ── Train / Val split ──────────────────────────────────────────────────
    torch.manual_seed(SEED)
    train_size = max(1, int(len(dataset) * TRAIN_SPLIT))
    val_size   = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    print(f"\n  Train samples : {train_size}")
    print(f"  Val   samples : {val_size}")

    # ── Model ──────────────────────────────────────────────────────────────
    model     = DrugInteractionGNN(
        node_features=39,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=30, gamma=0.5
    )

    # ── Training loop ──────────────────────────────────────────────────────
    print(f"\n  Training for {EPOCHS} epochs …\n")
    print(f"  {'Epoch':<8} {'Train Loss':<14} {'Val Loss':<12} {'Val Acc'}")
    print(f"  {'─'*8} {'─'*14} {'─'*12} {'─'*10}")

    best_val_acc  = 0.0
    best_val_loss = float("inf")

    train_list = list(train_data)
    val_list   = list(val_data)

    for epoch in range(1, EPOCHS + 1):
        train_loss          = train_epoch(model, train_list, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_list, device)
        scheduler.step()

        # Print every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            marker = " ◄ best" if val_acc > best_val_acc else ""
            print(f"  {epoch:<8} {train_loss:<14.4f} {val_loss:<12.4f} {val_acc:.1%}{marker}")

        # Save best model
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc  = val_acc
            best_val_loss = val_loss
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "val_acc"    : val_acc,
                "val_loss"   : val_loss,
                "hidden_dim" : HIDDEN_DIM,
                "num_classes": NUM_CLASSES,
            }, MODEL_PATH)

    # ── Final evaluation ───────────────────────────────────────────────────
    print(f"\n  Training complete!")
    print(f"  Best val accuracy : {best_val_acc:.1%}")
    print(f"  Model saved       → {MODEL_PATH}")

    # Load best model and show classification report
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    _, _, preds, actuals = evaluate(model, val_list, device)

    present_labels = sorted(set(actuals))
    target_names   = [LABEL_NAMES[i] for i in present_labels]

    print(f"\n  Classification Report (validation set):")
    print(f"  {'-'*45}")
    report = classification_report(
        actuals, preds,
        labels=present_labels,
        target_names=target_names,
        zero_division=0,
    )
    for line in report.split("\n"):
        print(f"  {line}")

    print(f"\n✅  Training done. Run app.py next to test predictions!\n")


if __name__ == "__main__":
    main()