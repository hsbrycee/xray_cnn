# VetXRay Project Overview

## Goal
This project analyzes veterinary chest X-ray images stored as DICOM files and links them with structured labels from the accompanying spreadsheet.

The current notebook workflow focuses on:
- loading all X-ray images into a PyTorch tensor
- aligning each image with metadata such as species, breed, projection, quality, and findings
- cleaning the metadata
- training baseline models for species classification

The longer-term goal is to analyze the radiographic findings in the X-ray images, not just identify the species. The `findings` field is already loaded into the notebook and can be used later for multi-label pathology prediction.

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

## Notebook Flow
The main notebook is [Image upload.ipynb](/d:/project/vetxray/Image%20upload.ipynb).

It currently walks through:
1. loading and preprocessing DICOM files
2. building a tensor dataset
3. inspecting missing metadata
4. cleaning and splitting the dataset
5. training a simple fully connected baseline
6. training a small CNN baseline

## Current Status
What is already in place:
- end-to-end DICOM loading
- metadata alignment
- species label encoding
- train/test split
- PyTorch dataset and dataloader setup

What is still natural to add next:
- multi-label encoding for `findings`
- pretrained CNN backbones such as DenseNet121
- confusion matrices and per-class metrics
- experiment tracking and model checkpointing

## Environment
Use a Python 3.11 environment with the packages listed in [requirements.txt]

Install with:

```bash
pip install -r requirements.txt
```
