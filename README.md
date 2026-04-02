# VetXRay Project Overview

## Goal
This project analyzes veterinary chest X-ray DICOM studies and links them to structured labels from the companion spreadsheet. The immediate objective is to build reliable pathology models from the `TAG` field, with a current focus on `cardiomegaly` prediction.

The repository currently supports:
- DICOM loading and normalization
- metadata alignment from the spreadsheet
- binary label creation for cardiomegaly
- baseline model training with DenseNet121

The broader objective is to move from simple image classification toward robust radiographic finding prediction with a cleaner validation workflow and more careful dataset controls.

## Data
The project expects:
- DICOM files under `RX_1/RX_1`
- metadata in `File list with tags.xlsx`

Each sample is represented by:
- `X`: image tensor with shape `[N, 1, H, W]`
- `metadata_df`: aligned metadata table

Important metadata fields:
- `species`
- `breed`
- `projection`
- `quality`
- `findings`
- `PatientName`

## Dataset Snapshot
The spreadsheet contains 9,973 studies.

Relevant label counts:
- `cardiomegaly`: 1,738 positives
- non-cardiomegaly: 8,235 negatives
- `no_finding`: 3,923
- `exclude`: 2,275

Quality distribution:
- `correct`: 6,916
- `positioning`: 1,542
- `exclude`: 1,154
- `underexposed`: 291
- `overexposed`: 69

Projection distribution:
- `LL`: 5,660
- `DV`: 3,231
- `VD`: 1,082

Patient leakage risk is high in the current raw dataset: 9,613 of 9,973 images belong to patients with more than one image, so image-level random splitting is likely to mix the same patient across train and test.

## Main Files
The main notebook is [Image upload.ipynb].

The current cardiomegaly training script is [train_cardiomegaly_densenet.py].

Utility code for DICOM preprocessing, target building, and training loops lives in [vetxray_bigheart_utils.py].

## Current Cardiomegaly Workflow
The binary cardiomegaly pipeline currently does the following:
1. load DICOM files and normalize each image independently to `[0, 1]`
2. resize images to `224 x 224`
3. parse spreadsheet tags into a `findings` list
4. create a binary label where `cardiomegaly == 1`
5. split the data with a random image-level train/test split
6. oversample the positive class with `WeightedRandomSampler`
7. train only the DenseNet121 classifier head for 5 epochs
8. choose the best threshold by sweeping the test set

## Big Heart Prediction Issue
The current model can train, but it still struggles to extract reliable signal from the dataset for big heart prediction. This should be treated as a workflow and dataset problem first, not just a model-capacity problem.

Observed reasons this can fail:
- The pipeline keeps poor-quality studies and many rows tagged `exclude`, which injects easy shortcuts and noisy negatives.
- The current split is image-level instead of patient-level, so evaluation can be unstable or misleading.
- Threshold tuning is done on the test set, which leaks evaluation information into model selection.
- Only the classifier head is trained, while the pretrained backbone stays frozen and may not adapt enough to veterinary grayscale radiographs.
- Five epochs is probably too short for a meaningful transfer-learning run.
- Cardiomegaly is not isolated from related thoracic findings; many positives co-occur with `alveolar_pattern`, `bronchial_pattern`, or `interstitial_pattern`, so the model may latch onto secondary pathology instead of cardiac silhouette size.
- Images are resized directly to `224 x 224`, which can distort anatomy and make cardiac size estimation harder.
- Per-image min-max normalization may remove useful study-to-study intensity structure.
- Metadata normalization is incomplete; for example, `Quality` contains both `correct` and `Correct`, and species names contain small inconsistencies.
- Projection is mixed in one binary model. Heart appearance differs across `LL`, `DV`, and `VD`, so the classifier may be learning projection cues along with disease cues.

## Recommended Workflow
The project workflow should be tightened before drawing conclusions about model performance.

Recommended sequence:
1. filter out rows with `quality in {"exclude", "overexposed"}` and decide explicitly whether `positioning` and `underexposed` should be kept
2. standardize metadata values such as `Quality`, `specie`, and tag spelling before label creation
3. split by `PatientName`, not by image, and create separate train, validation, and test sets
4. tune thresholds on the validation set only, then report final metrics once on the held-out test set
5. report precision, recall, F1, ROC-AUC, PR-AUC, and confusion matrices
6. inspect failure cases visually, especially false negatives on true cardiomegaly studies
7. compare a single mixed-projection model against projection-specific models
8. unfreeze part or all of the DenseNet backbone after warm-up and train longer with early stopping
9. add augmentation that preserves anatomy, such as mild rotation, translation, and contrast jitter, but avoid aggressive geometric distortion
10. keep experiment logs, saved checkpoints, and exact split files so runs are reproducible

## Immediate Next Improvements
Concrete next steps for this repository:
- add a validation split to `train_cardiomegaly_densenet.py`
- exclude low-quality or explicitly excluded studies before training
- switch to patient-level grouping for splits
- save the best model checkpoint based on validation F1 or PR-AUC
- export per-sample predictions for later error analysis
- test partially unfrozen DenseNet training instead of classifier-only training

## Environment
Use a Python 3.11 environment with the packages listed in [requirements.txt].

Install with:

```bash
pip install -r requirements.txt
```
