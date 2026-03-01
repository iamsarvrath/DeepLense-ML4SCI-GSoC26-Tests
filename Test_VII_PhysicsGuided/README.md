## Test VII – Physics-Guided ML (PINN)

This folder contains the solution for **Specific Test VII: Physics-Guided ML**.

### Goal

Build a PyTorch classifier for the 3-class lensing dataset that incorporates the **gravitational lens equation** (or related physics) into the architecture or loss to improve performance over the Common Test I baseline.

### Notebook

- `Test_VII_PhysicsGuided.ipynb` – implementation:
  - Reuse or adapt the baseline model from Common Test I.
  - Add physics-guided components (e.g. additional loss term, auxiliary outputs, or constrained layers).
  - Train with 90:10 train–validation split.
  - Compare **ROC curves** and **AUC** against the Common Test I model.

### Data

Expected path for the dataset:

- `../data/common/`

Uses the same `dataset.zip` contents as Common Test I.

### Outputs

- `outputs/` – saved metrics, comparison plots, and trained weights if needed.

