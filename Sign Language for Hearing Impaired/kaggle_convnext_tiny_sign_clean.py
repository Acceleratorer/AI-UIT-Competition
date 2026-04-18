import os
import re
import csv
import time
import math
import copy
import pickle
import random
import zipfile
from pathlib import Path
from collections import Counter
from contextlib import nullcontext

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
from torchvision import models

# =========================
# CONFIG
# =========================
SEED = 42
TARGET_FRAMES = 16
IMG_SIZE = 224
BATCH_SIZE = 8
NUM_EPOCHS = 16
LR = 2e-4
WEIGHT_DECAY = 1e-4
VAL_SIZE = 0.15
HIDDEN_SIZE = 320
PATIENCE = 5
NUM_WORKERS = 2
TTA_VIEWS = 3
USE_AMP = True

INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working")
EXTRACT_ROOT = WORK_ROOT / "sign_dataset_work"
BEST_MODEL_PATH = WORK_ROOT / "best_convnext_tiny_bigru.pth"
SUBMISSION_CSV = WORK_ROOT / "submission.csv"
SUBMISSION_ZIP = WORK_ROOT / "submission.zip"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Official torchvision classification backbone allowed by the competition rule.
WEIGHTS = models.ConvNeXt_Tiny_Weights.DEFAULT
IMAGENET_MEAN = np.array(WEIGHTS.transforms().mean, dtype=np.float32)
IMAGENET_STD = np.array(WEIGHTS.transforms().std, dtype=np.float32)


# =========================
# UTILS
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def strip_group(video_name: str) -> str:
    stem = Path(video_name).stem
    return re.sub(r"_\d+$", "", stem)


def resolve_dataset_root(search_root: Path):
    for mapping_path in search_root.rglob("label_mapping.pkl"):
        root = mapping_path.parent
        if (root / "train").exists() and (root / "test").exists():
            return root
    return None


def find_zip_candidate(search_root: Path):
    preferred = [
        "Dataset-20260328T164236Z-1-001.zip",
        "dataset.zip",
    ]
    for name in preferred:
        matches = list(search_root.rglob(name))
        if matches:
            return matches[0]

    for path in search_root.rglob("*.zip"):
        return path
    return None


def extract_nested_zip(src_zip: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src_zip, "r") as outer:
        names = outer.namelist()
        inner_zip_name = None
        for name in names:
            lower = name.lower()
            if lower.endswith("dataset.zip") or (lower.endswith(".zip") and "dataset" in lower):
                inner_zip_name = name
                break

        if inner_zip_name is None:
            outer.extractall(dst_dir)
        else:
            nested_bytes = outer.read(inner_zip_name)
            nested_path = dst_dir / "_inner_dataset.zip"
            nested_path.write_bytes(nested_bytes)
            with zipfile.ZipFile(nested_path, "r") as inner:
                inner.extractall(dst_dir)


def prepare_dataset():
    direct_root = resolve_dataset_root(INPUT_ROOT)
    if direct_root is not None:
        return direct_root

    extracted_root = resolve_dataset_root(EXTRACT_ROOT)
    if extracted_root is not None:
        return extracted_root

    zip_path = find_zip_candidate(INPUT_ROOT)
    if zip_path is None:
        raise FileNotFoundError(
            "Could not find dataset zip or extracted dataset under /kaggle/input. "
            "Add the dataset in Kaggle notebook settings first."
        )

    print(f"[INFO] Extracting dataset from: {zip_path}")
    extract_nested_zip(zip_path, EXTRACT_ROOT)

    extracted_root = resolve_dataset_root(EXTRACT_ROOT)
    if extracted_root is None:
        raise FileNotFoundError(
            "Extracted files but could not locate train/, test/, and label_mapping.pkl together."
        )
    return extracted_root


def scan_dataset(train_dir: Path, label_mapping: dict):
    items = []
    for label_name in sorted(os.listdir(train_dir)):
        class_dir = train_dir / label_name
        if not class_dir.is_dir() or label_name not in label_mapping:
            continue
        for video_name in sorted(os.listdir(class_dir)):
            if not video_name.lower().endswith(".mp4"):
                continue
            items.append(
                {
                    "path": class_dir / video_name,
                    "label_name": label_name,
                    "label_idx": label_mapping[label_name],
                    "group": strip_group(video_name),
                    "video_name": video_name,
                }
            )
    return items


def read_video(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
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


def temporal_sample(frames: np.ndarray, target_frames: int, view_idx: int = 0, num_views: int = 1) -> np.ndarray:
    num_frames = len(frames)
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


def train_augment(frames: np.ndarray, out_size: int = 224) -> np.ndarray:
    t, h, w, _ = frames.shape

    scale = random.uniform(0.90, 1.00)
    crop_h = max(16, int(h * scale))
    crop_w = max(16, int(w * scale))
    top = 0 if crop_h == h else random.randint(0, h - crop_h)
    left = 0 if crop_w == w else random.randint(0, w - crop_w)
    frames = frames[:, top : top + crop_h, left : left + crop_w, :]

    if random.random() < 0.55:
        alpha = random.uniform(0.9, 1.1)
        beta = random.uniform(-10, 10)
        frames = np.clip(frames.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    if random.random() < 0.12:
        frames = np.stack([cv2.GaussianBlur(f, (3, 3), 0) for f in frames], axis=0)

    frames = resize_frames(frames, out_size)
    return frames


def to_tensor(frames: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=x.dtype).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x


# =========================
# DATASETS
# =========================
class SignTrainDataset(Dataset):
    def __init__(self, items, target_frames=16, image_size=224, train_mode=False):
        self.items = items
        self.target_frames = target_frames
        self.image_size = image_size
        self.train_mode = train_mode

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        frames = read_video(item["path"])
        frames = temporal_sample(frames, target_frames=self.target_frames)
        frames = train_augment(frames, out_size=self.image_size) if self.train_mode else resize_frames(frames, size=self.image_size)
        frames = to_tensor(frames)
        return {
            "frames": frames,
            "label_idx": torch.tensor(item["label_idx"], dtype=torch.long),
            "label_name": item["label_name"],
            "path": str(item["path"]),
        }


class SignTestDataset(Dataset):
    def __init__(self, test_dir: Path, target_frames=16, image_size=224, view_idx=0, num_views=1):
        self.test_dir = Path(test_dir)
        self.files = sorted([f for f in os.listdir(self.test_dir) if f.lower().endswith(".mp4")])
        self.target_frames = target_frames
        self.image_size = image_size
        self.view_idx = view_idx
        self.num_views = num_views

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        video_name = self.files[idx]
        video_path = self.test_dir / video_name
        frames = read_video(video_path)
        frames = temporal_sample(
            frames,
            target_frames=self.target_frames,
            view_idx=self.view_idx,
            num_views=self.num_views,
        )
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
class ConvNeXtTinyBiGRU(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int = 320):
        super().__init__()
        backbone = models.convnext_tiny(weights=WEIGHTS)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.norm = backbone.classifier[0]
        feature_dim = 768

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
            nn.Dropout(0.35),
            nn.Linear(hidden_size * 2, num_classes),
        )

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        x = self.features(x)
        x = self.avgpool(x)
        x = self.norm(x)
        x = torch.flatten(x, 1)
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


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    y_true, y_pred = [], []

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
        "loss": float(np.mean(losses)) if losses else 0.0,
        "macro_f1": macro_f1(y_true, y_pred),
    }


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, amp_enabled=True):
    model.train()
    losses = []

    for batch in tqdm(loader, desc="train", leave=False):
        frames = batch["frames"].to(device, non_blocking=True)
        labels = batch["label_idx"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        amp_ctx = torch.cuda.amp.autocast(enabled=amp_enabled) if device.type == "cuda" else nullcontext()
        with amp_ctx:
            logits = model(frames)
            loss = criterion(logits, labels)

        if amp_enabled and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        losses.append(loss.item())

    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def predict_test(model, test_dir, idx_to_label, device, target_frames=16, image_size=224, batch_size=8, tta_views=3):
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
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_test,
        )

        view_logits = []
        view_names = []
        model.eval()
        for batch in tqdm(loader, desc=f"infer view {view_idx + 1}/{tta_views}", leave=False):
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
    return [(video_name, idx_to_label[pred]) for video_name, pred in zip(ordered_names, preds)]


# =========================
# MAIN
# =========================
def main():
    set_seed(SEED)

    print("CUDA available:", torch.cuda.is_available())
    print("DEVICE:", DEVICE)
    if DEVICE.type != "cuda":
        print("[WARN] GPU is not enabled. In Kaggle, set Accelerator = GPU.")

    dataset_root = prepare_dataset()
    train_dir = dataset_root / "train"
    test_dir = dataset_root / "test"
    mapping_path = dataset_root / "label_mapping.pkl"

    with open(mapping_path, "rb") as f:
        label_mapping = pickle.load(f)
    idx_to_label = {v: k for k, v in label_mapping.items()}

    items = scan_dataset(train_dir, label_mapping)
    labels = np.array([x["label_idx"] for x in items])
    groups = np.array([x["group"] for x in items])

    splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=SEED)
    train_idx, val_idx = next(splitter.split(np.zeros(len(items)), labels, groups))

    train_items = [items[i] for i in train_idx]
    val_items = [items[i] for i in val_idx]

    class_counts = Counter([x["label_idx"] for x in train_items])
    sample_weights = torch.tensor([1.0 / class_counts[x["label_idx"]] for x in train_items], dtype=torch.double)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_ds = SignTrainDataset(train_items, target_frames=TARGET_FRAMES, image_size=IMG_SIZE, train_mode=True)
    val_ds = SignTrainDataset(val_items, target_frames=TARGET_FRAMES, image_size=IMG_SIZE, train_mode=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        collate_fn=collate_train,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        collate_fn=collate_train,
    )

    print(
        f"[INFO] classes={len(label_mapping)} | train={len(train_items)} | "
        f"val={len(val_items)} | test={len(list(test_dir.glob('*.mp4')))}"
    )

    model = ConvNeXtTinyBiGRU(num_classes=len(label_mapping), hidden_size=HIDDEN_SIZE).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda" and USE_AMP))

    best_f1 = -1.0
    best_state = None
    bad_epochs = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        start = time.time()
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            DEVICE,
            amp_enabled=(DEVICE.type == "cuda" and USE_AMP),
        )
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
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, BEST_MODEL_PATH)
            bad_epochs = 0
            print(f"[OK] saved best model -> {BEST_MODEL_PATH} (macro_f1={best_f1:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print("[INFO] Early stopping triggered.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    model.load_state_dict(best_state)

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

    with open(SUBMISSION_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_name", "label"])
        writer.writerows(predictions)
    print(f"[OK] wrote {SUBMISSION_CSV} with {len(predictions)} rows")

    with zipfile.ZipFile(SUBMISSION_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(SUBMISSION_CSV, arcname="submission.csv")
    print(f"[OK] wrote {SUBMISSION_ZIP}")

    print("[DONE] Files in /kaggle/working:")
    for p in sorted(WORK_ROOT.glob("submission*")):
        print(" -", p)


if __name__ == "__main__":
    main()
