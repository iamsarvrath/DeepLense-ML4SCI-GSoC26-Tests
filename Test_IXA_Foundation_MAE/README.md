## Test IX.A – Foundation Model (Masked Autoencoder + Classification)

This folder contains the solution for **Specific Test IX.A: Foundation Model**.

### Goals

1. **Pretraining**: Train a **Masked Autoencoder (MAE)** on the `no_sub` samples only, to reconstruct masked portions of input images and learn a good latent representation for strong lensing images.
2. **Fine-tuning**: Take the pretrained encoder and add a classification head to perform **3-class classification**:
   - `no_sub`
   - `cdm`
   - `axion`

using a **90:10 train–validation split**.

### Notebook

- `Test_IXA_Foundation_MAE.ipynb` – implementation:
  - Data loading and preparation (no_sub-only for pretraining, full dataset for fine-tuning).
  - MAE architecture (encoder, decoder, masking strategy).
  - Pretraining loop on `no_sub`.
  - Fine-tuning of encoder + classifier head on full dataset.
  - Evaluation with **ROC curves** and **AUC**.

### Data

Expected path for the foundation dataset:

- `../data/foundation/`

Place the extracted contents of the foundation dataset (`no_sub`, `cdm`, `axion`) here (not committed to git).

### Outputs

- `outputs/` – saved pretrained weights, fine-tuned classifier weights, metrics, and plots.

