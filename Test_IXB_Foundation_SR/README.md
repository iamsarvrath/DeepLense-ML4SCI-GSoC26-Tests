## Test IX.B – Foundation Model Fine-Tuned for Super-Resolution

This folder contains the solution for **Specific Test IX.B: Foundation Model Super-Resolution**.

### Goal

Take the **pretrained model from Test IX.A** (specifically, the MAE encoder) and fine-tune it for an **image super-resolution** task:

- Input: Low-resolution (LR) strong lensing images.
- Output: High-resolution (HR) reconstructions.

You must report:

- **MSE (Mean Squared Error)**
- **SSIM (Structural Similarity Index)**
- **PSNR (Peak Signal-to-Noise Ratio)**

on a **90:10 train–validation split**.

### Notebook

- `Test_IXB_Foundation_SR.ipynb` – implementation:
  - Load the pretrained encoder weights from Test IX.A.
  - Attach/define a decoder or upsampling head for super-resolution.
  - Train on HR/LR pairs.
  - Evaluate and log MSE, SSIM, PSNR on validation data.

### Data

Expected path for the super-resolution dataset:

- `../data/sr/`

Place the extracted contents of the HR/LR dataset here (not committed to git).

### Outputs

- `outputs/` – saved SR model weights, metrics, and example reconstructions.

