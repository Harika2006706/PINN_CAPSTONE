# Physics-Informed Neural Network (PINN) for Digital Lensless Holographic Microscopy (DLHM)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Physics-Informed Deep Learning framework for continuous focal depth estimation ($z$), numerical diffraction refocusing via the Angular Spectrum Method (ASM), and biological specimen classification from raw out-of-focus digital holograms.

---

## 📌 Key Highlights
* **Zero-Leakage Generalization:** 5-Fold Group Cross-Validation partitioned strictly by axial planes ($z$) ensures no shared fields of view between training and test sets.
* **Angular Spectrum Method (ASM) Physics:** Differentiable wave propagation incorporates the Helmholtz wave equation transfer function $H(f_x, f_y; z) = \exp(i k_z z)$ directly into the neural network loss.
* **Continuous Focal Depth Regression:** Accurately recovers sample distance $z$ with sub-millimeter precision, reducing depth estimation error from $1.60\text{ mm}$ (baseline) down to $0.56\text{ mm}$ (PINN).
* **10-Class Specimen Identification:** Identifies complex biological and synthetic specimens (Cheek cells, Volvox, Spirogyra, Fly leg, Honeybee wing, Skeletal muscle, SM600 fiber, etc.) with **99.16% zero-leakage accuracy**.

---

## 📂 Repository Structure

```
├── configs/               # Hyperparameters, optical parameters, dataset configs
├── notebooks/             # Retrained PINN & benchmark Jupyter notebooks
│   └── PINN_RETRAINED.ipynb
├── scripts/               # Operational pipelines & evaluation suites
│   ├── run_full_single_inference_pipeline.py  # Master end-to-end inference
│   ├── evaluate_all_10_objects.py             # 10-class evaluation benchmark
│   └── train_10class_pinn.py                  # PINN training routine
├── src/                   # Core modular library
│   ├── models/            # Baseline CNN & Physics-Informed Neural Network architectures
│   ├── physics/           # Differentiable Angular Spectrum Method (ASM) operators
│   └── training/          # Loss functions & trainer implementations
├── tests/                 # Unit tests (ASM physics, model forward passes, datasets)
├── requirements.txt       # Environment dependencies
├── .gitignore             # Excludes large raw datasets (50+ GB) & binary weights
└── README.md              # Project documentation
```

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/<your-username>/PINN-DLHM.git
cd PINN-DLHM
pip install -r requirements.txt
```

### 2. Run End-to-End Inference
Execute the inference pipeline on any raw test hologram:
```bash
python scripts/run_full_single_inference_pipeline.py <path_to_raw_hologram.bmp>
```

**The pipeline executes the full physics chain:**
1. Loads ground-truth focal distance $z$ from metadata.
2. Predicts $z$-distance: Baseline CNN vs. Physics-Informed (PINN).
3. Backpropagates the optical wavefield $|U(x, y; z)|$ via ASM transfer function.
4. Identifies the specimen class with softmax confidence distribution.
5. Generates and saves a 3-panel scientific visualization figure.

---

## 📊 Experimental Results

| Metric | Baseline (Without Physics) | Proposed PINN (With ASM Physics) | Gain |
| :--- | :---: | :---: | :---: |
| **Focal Depth Accuracy ($z$)** | 87.47% | **95.62%** | **+8.16%** |
| **Physical Depth Error ($|\Delta z|$)** | 1.60 mm | **0.56 mm** | **-1.04 mm** |
| **Specimen Classification Accuracy** | 97.83% | **99.16%** | **+1.33%** |
| **Sensor Noise Robustness ($\sigma=0.10$)** | 88.60% | **96.50%** | **+7.90%** |

---

## 📜 Citation & License
This project is open-source under the MIT License.
