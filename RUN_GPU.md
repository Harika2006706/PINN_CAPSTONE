# Running DLHM Baseline on GPU

This guide explains how to quickly transfer and run the `DLHMBaselineCNN` training pipeline on external GPU instances like Kaggle or Google Colab.

## 1. Transfer Project Files
Zip this entire project repository. This must include:
*   `data/` (containing `development_manifest.csv` and the actual `.bmp` images)
*   `src/`
*   `configs/`
*   `scripts/`
*   `requirements.txt`

Upload and extract the zip file to your Kaggle/Colab notebook environment. 
*(Tip for Kaggle: Upload the zip as a private Kaggle Dataset to easily attach it to notebooks without waiting for uploads every session).*

## 2. Install Exact Dependencies
To prevent library version mismatches, install the pinned requirements:
```bash
!pip install -r requirements.txt
```

## 3. Run Training
Execute the training script from the root of the project repository. It will automatically detect the GPU (`cuda`) and begin processing the 50 epochs:
```bash
!python scripts/train_baseline.py
```

## 4. Retrieve Results
Once training finishes, all checkpoints and logs will be saved automatically to:
`results/baseline_training/`

You can download `best_baseline.pth` and `metrics_history.json` directly from the notebook output interface to review the metrics (MAE, RMSE, R²) and use the trained weights.
