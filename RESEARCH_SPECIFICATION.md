# RESEARCH SPECIFICATION

## 1. PROJECT OBJECTIVE
* **Problem**: Conventional reconstruction of inline Digital Holographic Microscopy (DHM) requires an exhaustive search to find the correct in-focus plane. This is computationally expensive and error-prone.
* **Intended Input**: Multi-plane holograms (or a single out-of-focus hologram) captured via inline DHM.
* **Intended Output**: A Physics-Informed Neural Network (PINN) that can predict the transformation of an out-of-focus hologram toward the in-focus plane, enabling direct classification or inference without explicitly searching for the focal plane first.

## 2. DATASET
* **Actual Dataset**: "Representative holograms" containing `1.bmp` through `12.bmp`, `ref.bmp`, `FN_Wing.png`, and `MW_510nm_Cheek_cell.jpg`. 
* **Image Properties**: 12 BMPs of size 4640 x 3506 pixels (uint8). Signal analysis confirms they are high-frequency interference patterns (raw holograms). They are perfectly laterally registered (0-pixel shift).
* **Labels / Ground Truth**: There are **no labels, no ground truth z-distances, and no reference in-focus phase/amplitude maps** provided for these 12 images.
* **Explicit Physical Parameters**: The filename `MW_510nm_Cheek_cell.jpg` suggests a wavelength of 510 nm. No other physical parameters are explicitly provided in the dataset.
* **Distinction from DLHM Databases**: The documented ODP-DLHM and FocusNET (UNAL) databases consist of different file formats (.jpg or no extension), explicit sample names (e.g., Fibra, USAF), and extensive labeling (thousands of images with calibrated $z$ distances and a point-source distance $L \approx 18.8$ mm). Our 12 `Representative holograms` do **not** match the filenames, structure, or metadata of those published databases. 

## 3. PHYSICS
* **Angular Spectrum Method (ASM)**: The ASM numerically simulates free-space wave propagation. In the proposed PINN, ASM directly links holograms recorded at different axial positions. The network uses the ASM as a physical constraint (loss function or layer) to enforce consistency between defocus planes.
* **Required Physical Parameters**:
  1. Wavelength ($\lambda$)
  2. Pixel pitch / Effective pixel size ($\Delta x, \Delta y$)
  3. Axial spacing between planes ($\Delta z$) or absolute reconstruction distances ($z$)
  4. Illumination type (Plane wave for inline DHM vs. Spherical for DLHM)
* **Documented Values in References**: FocusNET uses $\lambda = 632.8$ nm, $L = 18.8$ mm, and effective pixel pitch of 12.99 $\mu$m. The ODP-DLHM database uses $\lambda \in \{405, 510, 654, 633\}$ nm, $L \approx 18.8$ mm, and a Panasonic MN34230 camera (3.8 $\mu$m native pitch). 
* **Unknowns**: **All physical parameters for our 12 Representative holograms are completely unknown.** We do not know their $\lambda$, pixel pitch, $z$-spacing, or whether they are actually a $z$-stack or a time-series.

## 4. EXISTING DL BASELINE
* **FocusNET**: 
  * *Input*: A single 2-channel 256x256 tensor (raw DLHM hologram + its FFT amplitude).
  * *Architecture*: 5 Convolutional blocks (Conv2D, Batch Normalization, Max Pooling, Swish activation) followed by a regression head (Flatten, Dense layers).
  * *Target*: Ground truth in-focus reconstruction distance $z$.
  * *Loss / Metric*: Mean Squared Error (MSE) / Mean Absolute Error (MAE).
* **What to reuse**: The FFT amplitude preprocessing channel and the CNN feature extraction backbone are proven techniques for extracting holographic features.
* **What NOT to copy**: FocusNET predicts a single $z$-distance for a lensless setup (DLHM) using a purely data-driven supervised loss. Our project requires a physics-informed approach across *multiple* defocus planes in an inline DHM configuration, without relying on massive labeled $z$-distance ground truths.

## 5. PROPOSED MODEL
* **Approach**: A Physics-Informed Neural Network (PINN) (based on `pptEE74.pptx`).
* **Prediction**: The network should learn the transformation toward the in-focus plane, outputting a focused representation, classification, or inference directly from out-of-focus inputs.
* **Role of ASM**: ASM acts as the forward physical model. By propagating the network's prediction to different axial planes, the model's output can be directly compared to the raw out-of-focus holograms.
* **Physics Consistency**: Constraining the network with ASM ensures the learned representations obey wave propagation physics. This acts as a regularizer, making the model more robust to noise, blur, and varying defocus distances than a black-box CNN.

## 6. TRAINING AND EVALUATION
* **Dataset Split**: A traditional deep learning split (e.g., 80/20) is impossible with only 12 unlabelled holograms. If the PINN operates as an untrained physics-informed optimization (like phase retrieval) or requires self-supervised learning, the 12 planes act as the constraints for a single sample.
* **Target Variables**: The underlying complex wavefield (amplitude and phase) at the object plane.
* **Metrics**: Physics loss (MSE between ASM-propagated predictions and the 12 raw intensity holograms) and sharpness metrics (Variance of Laplacian).
* **Baselines**: Conventional autofocusing (Dubois, Sobel, Variance of Laplacian) or a purely data-driven black-box CNN.

## 7. RESEARCH CONSTRAINTS
* **No invented parameters**: We cannot implement the ASM until the exact $\lambda$, pixel pitch, and $\Delta z$ are definitively provided or numerically reverse-engineered.
* **Dataset Identity**: The 12 `Representative holograms` are assumed to be a standalone, unverified sequence. They must not be conflated with the ODP-DLHM or UNAL databases.
* **Validation Claims**: No claims of experimental validation can be made without ground truth data.
* **Fact vs. Assumption**: It is a *fact* that ASM requires physical parameters; it is an *assumption* that the 12 holograms are a $z$-stack (earlier analysis actually found no monotonic focus progression).
