"""Shared KT training loop (Section 3.2)."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def train_kt_model(
    model: nn.Module,
    train_dataset,
    val_dataset,
    model_name: str = "dkt",
    dataset_name: str = "ednet",
    lr: float = 1e-3,
    batch_size: int = 64,
    epochs: int = 100,
    patience: int = 10,
    device: str = "cpu",
    checkpoint_dir: str = "outputs/checkpoints",
):
    """
    Train a KT model with early stopping on validation AUC.
    Returns the best model and training history.
    """
    device = torch.device(device)
    model = model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{model_name}_{dataset_name}_best.pt"

    best_auc = 0.0
    no_improve = 0
    history = {"train_loss": [], "val_auc": []}

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_loss = 0
        n_batches = 0

        for batch in train_loader:
            q_ids = batch["q_ids"].to(device)
            kc_ids = batch["kc_ids"].to(device)
            corrects = batch["corrects"].to(device)
            mask = batch["mask"].to(device)

            # Model forward
            if hasattr(model, 'kc_embed'):  # AKT, SAINT
                logits = model(q_ids, kc_ids, corrects, mask)
            else:  # DKT
                logits_all = model(q_ids, corrects, mask)
                # Gather logits for the actual next question
                # Shift: predict step t+1 from state at step t
                logits = logits_all[:, :-1, :].gather(
                    2, q_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
                corrects_target = corrects[:, 1:]
                mask_target = mask[:, 1:]
                # Compute loss
                loss = criterion(logits, corrects_target)
                loss = (loss * mask_target).sum() / mask_target.sum().clamp(min=1)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
                continue

            # AKT/SAINT: response is already shifted inside the model
            # logits[t] predicts corrects[t] using history r[0..t-1]
            # No additional shift needed here
            loss = criterion(logits, corrects)
            loss = (loss * mask).sum() / mask.sum().clamp(min=1)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        history["train_loss"].append(avg_loss)

        # Validate
        val_auc = evaluate_kt(model, val_loader, device)
        history["val_auc"].append(val_auc)

        print(f"  Epoch {epoch:3d} | loss={avg_loss:.4f} | val_AUC={val_auc:.4f}"
              f" | best={best_auc:.4f} | patience={no_improve}/{patience}")

        if val_auc > best_auc:
            best_auc = val_auc
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Load best
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    print(f"  Best val AUC: {best_auc:.4f} | Saved: {ckpt_path}")
    return model, history


def evaluate_kt(model, loader, device):
    """Evaluate KT model, return AUC."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            q_ids = batch["q_ids"].to(device)
            kc_ids = batch["kc_ids"].to(device)
            corrects = batch["corrects"].to(device)
            mask = batch["mask"].to(device)

            if hasattr(model, 'kc_embed'):
                logits = model(q_ids, kc_ids, corrects, mask)
                # Response already shifted inside model: logits[t] predicts corrects[t]
                preds = torch.sigmoid(logits)
                valid = mask.bool()
                all_preds.extend(preds[valid].cpu().numpy())
                all_labels.extend(corrects[valid].cpu().numpy())
            else:
                logits_all = model(q_ids, corrects, mask)
                logits = logits_all[:, :-1, :].gather(
                    2, q_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
                preds = torch.sigmoid(logits)
                mask_target = mask[:, 1:]
                valid = mask_target.bool()
                all_preds.extend(preds[valid].cpu().numpy())
                all_labels.extend(corrects[:, 1:][valid].cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    if len(np.unique(all_labels)) < 2:
        return 0.5

    return roc_auc_score(all_labels, all_preds)
