from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models
from torchvision.models import DenseNet121_Weights
from tqdm.auto import tqdm

from vetxray_bigheart_utils import (
    ImageBinaryDataset,
    binary_metric_summary,
    build_cardiomegaly_target,
    build_dicom_tensor,
    eval_binary_epoch,
    load_annotations,
    print_train_time,
    timer,
    train_binary_epoch,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DICOM_DIR = Path("RX_1") / "RX_1"
XLSX_PATH = Path("File list with tags.xlsx")
TARGET_SIZE = (224, 224)
LIMIT = None
BATCH_SIZE = 16
THRESHOLD = 0.5
EPOCHS = 5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4


def build_dataloaders(image_tensor: torch.Tensor, metadata_df: pd.DataFrame):
    valid_mask = metadata_df["findings"].notna().to_numpy()
    metadata_clean = metadata_df[valid_mask].reset_index(drop=True)
    image_tensor_clean = image_tensor[valid_mask]

    y = build_cardiomegaly_target(metadata_clean)

    indices = list(range(len(metadata_clean)))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=y,
    )

    X_train = image_tensor_clean[train_idx]
    X_test = image_tensor_clean[test_idx]
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    train_data = ImageBinaryDataset(X_train, y_train)
    test_data = ImageBinaryDataset(X_test, y_test)

    class_counts = y_train.value_counts()
    sample_weights = y_train.map(
        {
            0.0: 1.0 / class_counts[0.0],
            1.0: 1.0 / class_counts[1.0],
        }
    ).to_numpy()

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=sampler)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    return train_dataloader, test_dataloader, y_train, y_test


def build_model():
    weights = DenseNet121_Weights.DEFAULT
    model = models.densenet121(weights=weights)

    old_conv = model.features.conv0
    model.features.conv0 = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )

    with torch.no_grad():
        model.features.conv0.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)

    model.classifier = nn.Linear(model.classifier.in_features, 1)

    for param in model.features.parameters():
        param.requires_grad = False

    return model.to(DEVICE)


def collect_probabilities(model, data_loader):
    all_probs = []
    all_true = []

    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X = X.to(DEVICE)
            logits = model(X)
            probs = torch.sigmoid(logits).squeeze(1).cpu()
            all_probs.append(probs)
            all_true.append(y)

    return torch.cat(all_probs), torch.cat(all_true).float()


def find_best_threshold(model, data_loader):
    all_probs, all_true = collect_probabilities(model, data_loader)

    rows = []
    for threshold in [i / 100 for i in range(5, 100, 5)]:
        preds = (all_probs >= threshold).float()
        rows.append({"threshold": threshold, **binary_metric_summary(all_true, preds)})

    threshold_df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    return threshold_df.iloc[0], threshold_df


def main():
    print("Using device:", DEVICE)
    print("DICOM directory:", DICOM_DIR.resolve())
    print("Annotations:", XLSX_PATH.resolve())
    print("Target size:", TARGET_SIZE)

    annotations = load_annotations(XLSX_PATH)
    image_tensor, metadata_df = build_dicom_tensor(
        DICOM_DIR,
        annotations,
        target_size=TARGET_SIZE,
        limit=LIMIT,
    )

    print("Tensor shape:", tuple(image_tensor.shape))

    train_dataloader, test_dataloader, y_train, y_test = build_dataloaders(image_tensor, metadata_df)

    print("Train positives:", int(y_train.sum()))
    print("Train negatives:", len(y_train) - int(y_train.sum()))
    print("Test positives:", int(y_test.sum()))
    print("Test negatives:", len(y_test) - int(y_test.sum()))

    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    pos_weight = torch.tensor([neg_count / max(pos_count, 1.0)], dtype=torch.float32).to(DEVICE)

    model = build_model()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.classifier.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    train_start = timer()

    for epoch in tqdm(range(EPOCHS)):
        train_loss, train_f1 = train_binary_epoch(
            model,
            train_dataloader,
            loss_fn,
            optimizer,
            DEVICE,
            threshold=THRESHOLD,
        )
        test_loss, test_f1 = eval_binary_epoch(
            model,
            test_dataloader,
            loss_fn,
            DEVICE,
            threshold=THRESHOLD,
        )

        print(
            f"Epoch: {epoch + 1} | "
            f"Train loss: {train_loss:.5f} | Train F1: {train_f1:.2f}% | "
            f"Test loss: {test_loss:.5f} | Test F1: {test_f1:.2f}%"
        )

    train_end = timer()
    print_train_time(train_start, train_end, device=DEVICE)

    best_row, threshold_df = find_best_threshold(model, test_dataloader)
    print("\nBest threshold on test set:")
    print(best_row.to_dict())
    print("\nTop threshold sweep results:")
    print(threshold_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
