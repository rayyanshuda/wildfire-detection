"""Training loop and prediction. Every model arm runs this identical code —
that is what makes the benchmark a fair comparison rather than a claim."""
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, roc_curve, log_loss

from data import WildfireDataset, train_tf, eval_tf, CSV, IMGS

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def predict(model, loader):
    """Return (y_true, y_score) as numpy. The only torch-aware piece."""
    model.eval()
    ys, ss = [], []
    with torch.no_grad():
        for xb, yb in loader:
            ss.append(torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy())
            ys.append(yb.numpy())
    return np.concatenate(ys), np.concatenate(ss)


def pick_threshold(y_true, y_score):
    """Youden's J: maximise (sensitivity + specificity - 1). Chosen on VAL only."""
    fpr, tpr, thr = roc_curve(y_true, y_score)
    t = float(thr[np.argmax(tpr - fpr)])
    return t if np.isfinite(t) else 0.5


def train_model(make_model, seed, epochs=40, lr=1e-3, patience=8, bs=32,
                tag='model', freeze_epochs=0, train_idx=None, val_idx=None):
    """train_idx: explicit list of training indices (stratified subsets for the
    data-efficiency experiment). val_idx: only for fast dry runs — real runs use
    the FULL val set at every training size, or the sizes aren't comparable."""
    torch.manual_seed(seed); np.random.seed(seed)

    train_ds = WildfireDataset(CSV, IMGS, 'train', train_tf)
    val_ds   = WildfireDataset(CSV, IMGS, 'val',   eval_tf)
    if train_idx is not None:
        train_ds = Subset(train_ds, list(train_idx))
    if val_idx is not None:
        val_ds = Subset(val_ds, list(val_idx))

    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=2,
                          pin_memory=True, drop_last=len(train_ds) > bs)
    val_dl   = DataLoader(val_ds, batch_size=64, shuffle=False,
                          num_workers=2, pin_memory=True)

    model     = make_model().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()

    def build_optim(phase):
        if phase == 'frozen':
            for p in model.backbone_parameters(): p.requires_grad = False
            opt, T = torch.optim.Adam(model.head_parameters(), lr=lr), freeze_epochs
        elif phase == 'unfrozen':
            for p in model.backbone_parameters(): p.requires_grad = True
            opt = torch.optim.Adam([
                {'params': model.backbone_parameters(), 'lr': lr / 100},
                {'params': model.head_parameters(),     'lr': lr / 10},
            ])
            T = epochs - freeze_epochs
        else:
            opt, T = torch.optim.Adam(model.parameters(), lr=lr), epochs
        return opt, torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(T, 1))

    optimizer, scheduler = build_optim('frozen' if freeze_epochs > 0 else 'single')
    best_auc, best_state, best_epoch, bad = -1, None, -1, 0
    history = []

    for epoch in range(epochs):
        if freeze_epochs > 0 and epoch == freeze_epochs:
            optimizer, scheduler = build_optim('unfrozen')
            print('  --- backbone unfrozen ---')

        model.train()
        run_loss, n = 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * len(yb); n += len(yb)
        scheduler.step()

        yt, ys   = predict(model, val_dl)
        val_loss = log_loss(yt, np.clip(ys, 1e-7, 1 - 1e-7), labels=[0, 1])
        val_auc  = roc_auc_score(yt, ys)
        history.append({'epoch': epoch, 'train_loss': run_loss / n,
                        'val_loss': val_loss, 'val_auc': val_auc,
                        'lr': scheduler.get_last_lr()[0]})
        print(f'  ep {epoch:2d}  train {run_loss/n:.4f}  val {val_loss:.4f}  auc {val_auc:.4f}')

        if val_auc > best_auc:
            best_auc, best_epoch, bad = val_auc, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                print(f'  early stop at {epoch}; best epoch was {best_epoch}')
                break

    model.load_state_dict(best_state)
    torch.save(best_state, f'/kaggle/working/{tag}_seed{seed}.pt')
    pd.DataFrame(history).to_csv(f'/kaggle/working/{tag}_seed{seed}_history.csv', index=False)
    yt_v, ys_v = predict(model, val_dl)
    return model, pd.DataFrame(history), best_auc, best_epoch, pick_threshold(yt_v, ys_v)
