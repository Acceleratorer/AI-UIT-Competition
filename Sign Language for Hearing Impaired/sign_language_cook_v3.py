import os
import re
import csv
import math
import time
import json
import random
import pickle
import zipfile
from collections import Counter

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_recall_fscore_support
from tqdm.auto import tqdm

from torchvision import models


# =========================
# CONFIG
# =========================
RAW_ZIP_PATH = "Dataset-20260328T164236Z-1-001.zip"   # outer zip OR inner dataset.zip
WORK_DIR = "dataset_work"
TARGET_FRAMES = 24
IMG_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 18
LR = 3e-4
WEIGHT_DECAY = 1e-4
VAL_SIZE = 0.15
HIDDEN_SIZE = 256
PATIENCE = 5
NUM_WORKERS = 2
SEED = 42
TTA_VIEWS = 3
USE_AMP = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# UTILS
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_dataset(raw_zip_path: str, work_dir: str):
    os.makedirs(work_dir, exist_ok=True)
    train_dir = os.path.join(work_dir, "train")
    test_dir = os.path.join(work_dir, "test")
    mapping_path = os.path.join(work_dir, "label_mapping.pkl")
    if os.path.exists(train_dir) and os.path.exists(test_dir) and os.path.exists(mapping_path):
        print("[OK] Dataset already prepared.")
        return train_dir, test_dir, mapping_path

    print(f"[INFO] Preparing dataset from: {raw_zip_path}")
    with zipfile.ZipFile(raw_zip_path, "r") as zf:
        names = zf.namelist()
        inner_zip = None
        for name in names:
            if name.lower().endswith("dataset.zip"):
                inner_zip = name
                break

        if inner_zip is None:
            print("[INFO] Detected direct dataset.zip structure. Extracting...")
            zf.extractall(work_dir)
        else:
            print(f"[INFO] Detected nested zip: {inner_zip}")
            nested_bytes = zf.read(inner_zip)
            nested_path = os.path.join(work_dir, "_inner_dataset.zip")
            with open(nested_path, "wb") as f:
                f.write(nested_bytes)
            with zipfile.ZipFile(nested_path, "r") as inner:
                inner.extractall(work_dir)

    assert os.path.exists(train_dir), f"Missing {train_dir}"
    assert os.path.exists(test_dir), f"Missing {test_dir}"
    assert os.path.exists(mapping_path), f"Missing {mapping_path}"
    return train_dir, test_dir, mapping_path


def strip_group(video_name: str) -> str:
    stem = os.path.splitext(os.path.basename(video_name))[0]
    return re.sub(r"_\d+$", "", stem)


def read_video(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"Cannot read frames from {video_path}")
    return np.stack(frames, axis=0)


def sample_frames(frames: np.ndarray, target_frames: int = 24, view_idx: int = 0, num_views: int = 1) -> np.ndarray:
    num_frames = frames.shape[0]
    if num_frames <= target_frames:
        idx = np.arange(num_frames)
        if num_frames < target_frames:
            pad = np.full(target_frames - num_frames, num_frames - 1)
            idx = np.concatenate([idx, pad])
        return frames[idx]

    if num_views <= 1:
        idx = np.linspace(0, num_frames - 1, target_frames).round().astype(int)
        return frames[idx]

    max_start = max(0, num_frames - target_frames)
    starts = np.linspace(0, max_start, num_views).round().astype(int)
    start = int(starts[min(view_idx, len(starts) - 1)])
    idx = np.linspace(start, start + target_frames - 1, target_frames).round().astype(int)
    idx = np.clip(idx, 0, num_frames - 1)
    return frames[idx]


def resize_frames(frames: np.ndarray, size: int = 224) -> np.ndarray:
    out = [cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA) for frame in frames]
    return np.stack(out, axis=0)


def augment_frames(frames: np.ndarray, out_size: int = 224) -> np.ndarray:
    t, h, w, c = frames.shape

    # spatial crop-jitter without flipping
    scale = random.uniform(0.88, 1.00)
    crop_h = max(16, int(h * scale))
    crop_w = max(16, int(w * scale))
    top = 0 if crop_h == h else random.randint(0, h - crop_h)
    left = 0 if crop_w == w else random.randint(0, w - crop_w)
    frames = frames[:, top:top + crop_h, left:left + crop_w, :]

    # light brightness/contrast jitter
    if random.random() < 0.6:
        alpha = random.uniform(0.9, 1.1)
        beta = random.uniform(-12, 12)
        frames = np.clip(frames.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # occasional blur
    if random.random() < 0.15:
        frames = np.stack([cv2.GaussianBlur(f, (3, 3), 0) for f in frames], axis=0)

    frames = resize_frames(frames, out_size)
    return frames


def to_tensor(frames: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x


def scan_train_items(train_dir: str, label_mapping: dict):
    items = []
    for label_name in sorted(os.listdir(train_dir)):
        class_dir = os.path.join(train_dir, label_name)
        if not os.path.isdir(class_dir):
            continue
        if label_name not in label_mapping:
            continue
        for video_name in sorted(os.listdir(class_dir)):
            if not video_name.lower().endswith(".mp4"):
                continue
            path = os.path.join(class_dir, video_name)
            items.append(
                {
                    "path": path,
                    "label_name": label_name,
                    "label_idx": label_mapping[label_name],
                    "group": strip_group(video_name),
                }
            )
    return items


# =========================
# DATASETS
# =========================
class SignTrainDataset(Dataset):
    def __init__(self, items, target_frames=24, image_size=224, train_mode=False):
        self.items = items
        self.target_frames = target_frames
        self.image_size = image_size
        self.train_mode = train_mode

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        frames = read_video(item["path"])
        frames = sample_frames(frames, target_frames=self.target_frames)
        if self.train_mode:
            frames = augment_frames(frames, out_size=self.image_size)
        else:
            frames = resize_frames(frames, size=self.image_size)
        frames = to_tensor(frames)
        return {
            "frames": frames,
            "label_idx": torch.tensor(item["label_idx"], dtype=torch.long),
            "label_name": item["label_name"],
            "path": item["path"],
        }


class SignTestDataset(Dataset):
    def __init__(self, test_dir, target_frames=24, image_size=224, view_idx=0, num_views=1):
        self.test_dir = test_dir
        self.files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(".mp4")])
        self.target_frames = target_frames
        self.image_size = image_size
        self.view_idx = view_idx
        self.num_views = num_views

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        video_name = self.files[idx]
        video_path = os.path.join(self.test_dir, video_name)
        frames = read_video(video_path)
        frames = sample_frames(frames, target_frames=self.target_frames, view_idx=self.view_idx, num_views=self.num_views)
        frames = resize_frames(frames, size=self.image_size)
        frames = to_tensor(frames)
        return {"frames": frames, "video_name": video_name}


def collate_train(batch):
    frames = torch.stack([b["frames"] for b in batch], dim=0)
    labels = torch.stack([b["label_idx"] for b in batch], dim=0)
    return {"frames": frames, "label_idx": labels}


def collate_test(batch):
    frames = torch.stack([b["frames"] for b in batch], dim=0)
    names = [b["video_name"] for b in batch]
    return {"frames": frames, "video_name": names}


# =========================
# MODEL
# =========================
class ResNetBiGRU(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int = 256):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        feature_dim = 512

        self.rnn = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(0.30),
            nn.Linear(hidden_size * 2, num_classes),
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        x = x.view(b, t, -1)
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        return self.head(x)


# =========================
# TRAINING
# =========================
def macro_f1(y_true, y_pred):
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return float(f1)


def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="valid", leave=False):
            frames = batch["frames"].to(device, non_blocking=True)
            labels = batch["label_idx"].to(device, non_blocking=True)

            logits = model(frames)
            loss = criterion(logits, labels)
            losses.append(loss.item())

            preds = logits.argmax(dim=1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    return {
        "loss": float(np.mean(losses)),
        "macro_f1": macro_f1(y_true, y_pred),
    }


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, amp_enabled=True):
    model.train()
    losses = []

    for batch in tqdm(loader, desc="train", leave=False):
        frames = batch["frames"].to(device, non_blocking=True)
        labels = batch["label_idx"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(frames)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())

    return float(np.mean(losses))


def predict_test(model, test_dir, idx_to_label, device, target_frames=24, image_size=224, batch_size=16, tta_views=3):
    all_logits = None
    ordered_names = None

    for view_idx in range(tta_views):
        dataset = SignTestDataset(
            test_dir=test_dir,
            target_frames=target_frames,
            image_size=image_size,
            view_idx=view_idx,
            num_views=tta_views,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_test,
        )

        view_logits = []
        view_names = []

        model.eval()
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"infer view {view_idx+1}/{tta_views}", leave=False):
                frames = batch["frames"].to(device, non_blocking=True)
                logits = model(frames).cpu()
                view_logits.append(logits)
                view_names.extend(batch["video_name"])

        view_logits = torch.cat(view_logits, dim=0)

        if all_logits is None:
            all_logits = view_logits
            ordered_names = view_names
        else:
            all_logits += view_logits

    preds = all_logits.argmax(dim=1).numpy().tolist()
    result = [(video_name, idx_to_label[pred]) for video_name, pred in zip(ordered_names, preds)]
    return result


def main():
    set_seed(SEED)
    train_dir, test_dir, mapping_path = prepare_dataset(RAW_ZIP_PATH, WORK_DIR)

    with open(mapping_path, "rb") as f:
        label_mapping = pickle.load(f)

    idx_to_label = {v: k for k, v in label_mapping.items()}
    items = scan_train_items(train_dir, label_mapping)

    labels = np.array([x["label_idx"] for x in items])
    groups = np.array([x["group"] for x in items])

    splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=SEED)
    train_idx, val_idx = next(splitter.split(np.zeros(len(items)), labels, groups))

    train_items = [items[i] for i in train_idx]
    val_items = [items[i] for i in val_idx]

    train_ds = SignTrainDataset(train_items, target_frames=TARGET_FRAMES, image_size=IMG_SIZE, train_mode=True)
    val_ds = SignTrainDataset(val_items, target_frames=TARGET_FRAMES, image_size=IMG_SIZE, train_mode=False)

    class_counts = Counter([x["label_idx"] for x in train_items])
    sample_weights = torch.tensor([1.0 / class_counts[x["label_idx"]] for x in train_items], dtype=torch.double)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_train,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_train,
    )

    print(f"[INFO] classes={len(label_mapping)} | train={len(train_items)} | val={len(val_items)} | test={len(os.listdir(test_dir))}")

    model = ResNetBiGRU(num_classes=len(label_mapping), hidden_size=HIDDEN_SIZE).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=(torch.cuda.is_available() and USE_AMP))

    best_f1 = -1.0
    bad_epochs = 0
    best_path = "best_model.pth"

    for epoch in range(1, NUM_EPOCHS + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE, amp_enabled=(torch.cuda.is_available() and USE_AMP))
        val_metrics = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        elapsed = time.time() - start
        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} | time={elapsed:.1f}s"
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            bad_epochs = 0
            torch.save(model.state_dict(), best_path)
            print(f"[OK] saved best model -> {best_path} (macro_f1={best_f1:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print("[INFO] Early stopping triggered.")
                break

    model.load_state_dict(torch.load(best_path, map_location=DEVICE))

    predictions = predict_test(
        model=model,
        test_dir=test_dir,
        idx_to_label=idx_to_label,
        device=DEVICE,
        target_frames=TARGET_FRAMES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        tta_views=TTA_VIEWS,
    )

    out_csv = "submission.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_name", "label"])
        writer.writerows(predictions)
    print(f"[OK] wrote {out_csv} with {len(predictions)} rows")

    out_zip = "submission.zip"
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_csv, arcname="submission.csv")
    print(f"[OK] wrote {out_zip}")


if __name__ == "__main__":
    main()