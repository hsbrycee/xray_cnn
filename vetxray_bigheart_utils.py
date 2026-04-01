from pathlib import Path
from timeit import default_timer as timer

import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


def load_annotations(xlsx_path: Path) -> pd.DataFrame:
    annotations = pd.read_excel(xlsx_path, engine="openpyxl")
    annotations["FileName"] = annotations["FileName"].astype(str)
    return annotations.set_index("FileName", drop=False)


def preprocess_dicom(path: Path, target_size=None):
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype("float32")

    p_low, p_high = img.min(), img.max()
    if p_high <= p_low:
        img = img * 0.0
    else:
        img = (img - p_low) / (p_high - p_low)

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        img = 1.0 - img

    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)

    if target_size is not None and tuple(img.shape) != tuple(target_size):
        tensor = F.interpolate(
            tensor,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

    metadata = {
        "filename": path.name,
        "original_height": int(img.shape[0]),
        "original_width": int(img.shape[1]),
        "modality": getattr(ds, "Modality", "N/A"),
        "photometric_interpretation": getattr(ds, "PhotometricInterpretation", "N/A"),
    }
    return tensor.squeeze(0), metadata


def build_dicom_tensor(dicom_dir: Path, annotations_indexed: pd.DataFrame, target_size=None, limit=None):
    dcm_paths = sorted(dicom_dir.rglob("*.dcm"))
    if limit is not None:
        dcm_paths = dcm_paths[:limit]

    if not dcm_paths:
        raise ValueError(f"No DICOM files found under {dicom_dir.resolve()}")

    tensors = []
    records = []

    for idx, path in enumerate(dcm_paths, start=1):
        image_tensor, dicom_meta = preprocess_dicom(path, target_size=target_size)
        tensors.append(image_tensor)

        record = dicom_meta.copy()
        if path.name in annotations_indexed.index:
            ann = annotations_indexed.loc[path.name]
            findings = [tag.strip() for tag in str(ann["TAG"]).split("|") if tag.strip()]
            record.update(
                {
                    "species": ann.get("specie", None),
                    "breed": ann.get("breed", None),
                    "projection": ann.get("Projection", None),
                    "quality": ann.get("Quality", None),
                    "findings": findings,
                }
            )
        else:
            record.update(
                {
                    "species": None,
                    "breed": None,
                    "projection": None,
                    "quality": None,
                    "findings": [],
                }
            )

        records.append(record)

        if idx % 250 == 0 or idx == len(dcm_paths):
            print(f"Processed {idx}/{len(dcm_paths)} DICOM files")

    image_tensor = torch.stack(tensors, dim=0).to(torch.float32)
    image_tensor = torch.nan_to_num(image_tensor, nan=0.0, posinf=1.0, neginf=0.0)
    metadata_df = pd.DataFrame(records)
    return image_tensor, metadata_df


def build_cardiomegaly_target(metadata_df: pd.DataFrame) -> pd.Series:
    return metadata_df["findings"].apply(
        lambda findings: 1.0 if "cardiomegaly" in findings else 0.0
    ).reset_index(drop=True)


class ImageBinaryDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: pd.Series):
        assert len(images) == len(labels), "images and labels must have the same length"
        self.images = images
        self.labels = labels.reset_index(drop=True)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx]
        y = torch.tensor(self.labels.iloc[idx], dtype=torch.float32)
        return x, y


class LinearBaseline(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int = 16):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_shape, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1),
        )

    def forward(self, x):
        return self.layer_stack(x)


class XRayCNN(nn.Module):
    def __init__(self, input_channels: int = 1, hidden_units: int = 16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_units, hidden_units * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units * 2, hidden_units * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_units * 2, hidden_units * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units * 4, hidden_units * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def binary_accuracy_fn(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return (y_true == y_pred).float().mean().item() * 100


def binary_f1_fn(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> float:
    y_true = y_true.float()
    y_pred = y_pred.float()

    tp = ((y_true == 1) & (y_pred == 1)).sum().item()
    fp = ((y_true == 0) & (y_pred == 1)).sum().item()
    fn = ((y_true == 1) & (y_pred == 0)).sum().item()

    precision = tp / max(tp + fp, eps)
    recall = tp / max(tp + fn, eps)
    f1 = 2 * precision * recall / max(precision + recall, eps)
    return f1 * 100


def binary_metric_summary(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> dict:
    y_true = y_true.float()
    y_pred = y_pred.float()

    tp = ((y_true == 1) & (y_pred == 1)).sum().item()
    tn = ((y_true == 0) & (y_pred == 0)).sum().item()
    fp = ((y_true == 0) & (y_pred == 1)).sum().item()
    fn = ((y_true == 1) & (y_pred == 0)).sum().item()

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, eps)
    recall = tp / max(tp + fn, eps)
    f1 = 2 * precision * recall / max(precision + recall, eps)

    return {
        "accuracy": accuracy * 100,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def train_binary_epoch(model, data_loader, loss_fn, optimizer, device, threshold=0.2):
    model.train()
    train_loss = 0.0
    train_f1 = 0.0

    for X, y in data_loader:
        X = X.to(device)
        y = y.float().to(device).unsqueeze(1)

        logits = model(X)
        loss = loss_fn(logits, y)
        preds = (torch.sigmoid(logits) >= threshold).float()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()
        train_f1 += binary_f1_fn(y, preds)

    return train_loss / len(data_loader), train_f1 / len(data_loader)


def eval_binary_epoch(model, data_loader, loss_fn, device, threshold=0.2):
    model.eval()
    test_loss = 0.0
    test_f1 = 0.0

    with torch.inference_mode():
        for X, y in data_loader:
            X = X.to(device)
            y = y.float().to(device).unsqueeze(1)

            logits = model(X)
            preds = (torch.sigmoid(logits) >= threshold).float()

            test_loss += loss_fn(logits, y).item()
            test_f1 += binary_f1_fn(y, preds)

    return test_loss / len(data_loader), test_f1 / len(data_loader)


def print_train_time(start, end, device=None):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time


__all__ = [
    "ImageBinaryDataset",
    "LinearBaseline",
    "XRayCNN",
    "binary_accuracy_fn",
    "binary_f1_fn",
    "binary_metric_summary",
    "build_cardiomegaly_target",
    "build_dicom_tensor",
    "eval_binary_epoch",
    "load_annotations",
    "preprocess_dicom",
    "print_train_time",
    "timer",
    "train_binary_epoch",
]
