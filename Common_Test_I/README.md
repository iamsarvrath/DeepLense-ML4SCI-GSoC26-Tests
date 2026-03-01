# Common Test I: Strong Lensing Substructure Classification

This is my baseline solution for the first common test. The goal here was to take high-resolution images of gravitational lenses and figure out if they contain any "extra" substructures—like dark matter subhalos or vortex disturbances—or if they're just plain lenses with no substructure at all.

### How I tackled it (My Strategy)

1.  **Keeping the details sharp**: Standard AI models (like a vanilla ResNet) tend to "blur" out small details early on because they use aggressive downsampling. Since the physics we're looking for is hidden in tiny, pixel-scale distortions, I removed the initial `MaxPool` layer. This keeps the early features at high resolution so we don't miss the subtle signs of dark matter.
2.  **Going Grayscale**: These lensing images are 1-channel grayscale, not full color. I adapted the first layer of the model by summing up the pre-trained ImageNet weights. This lets the model use its pre-trained "edge-detection" skills while being natively compatible with the scientific data.
3.  **Spinning the Rings**: Gravity doesn't care about orientation! A lensing ring could be tilted any which way. To make the model robust, I implemented random 90-degree rotations and flips. This forces the model to learn the actual *physics* of the disturbances, rather than just memorizing a specific angle.
4.  **The Gold Standard (AUC)**: Accuracy alone can be misleading. That's why I used **ROC-AUC** as the primary metric. It’s the standard in astrophysics classification because it measures how well the model separates classes across all thresholds, giving us a much more honest view of performance.

### What's inside?

- **[Common_Test_I.ipynb](file:///d:/tests/DeepLense-ML4SCI-GSoC26-Tests/Common_Test_I/Common_Test_I.ipynb)**: My end-to-end implementation from data loading to evaluation.
- **Automated Saving**: I set up the training loop to be "smart"—it automatically saves the best model weights to the `../model/` folder whenever it hits a new breakthrough in validation accuracy.
- **Inference Demo**: I added a section at the end so you can load the saved `.pth` file and see the model predict a new image instantly.

### The Results

The model performed exceptionally well, achieving:
- **Macro AUC**: 0.989+ (Nearly perfect class separation)
- **Validation Accuracy**: Consistent high performance (~94%+).

```text
--- Detailed Classification Report ---
                 precision    recall  f1-score   support

No Substructure       0.92      0.96      0.94      1250
Subhalo Substructure  0.94      0.92      0.93      1250
Vortex Substructure   0.96      0.94      0.95      1250

       accuracy                           0.94      3750
      macro avg       0.94      0.94      0.94      3750
   weighted avg       0.94      0.94      0.94      3750

Macro AUC: 0.9897
Weighted AUC: 0.9897
```

![Final Evaluation Results](./outputs/final_evaluation.png)

### How to run it

1.  **Data**: Place the dataset in `../data/common/`.
2.  **Environment**: Install dependencies from the root `requirements.txt`.
3.  **Run**: Just execute the notebook. The best model will be saved to `../model/common_test_i_best.pth`, and plots will be saved in `outputs/`.

