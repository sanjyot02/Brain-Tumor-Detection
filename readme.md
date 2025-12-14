
# Brain Tumor Classification using CNN

This project is a deep learning–based approach for automatic brain tumor detection and classification from MRI images using a Convolutional Neural Network (CNN).

The model classifies MRI scans into Glioma, Meningioma, Pituitary Tumor, or No Tumor and also provides visual explanation using Grad-CAM.

## Dataset

**Brain Tumor MRI Dataset** ([Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset))

**Classes:** Glioma, Meningioma, Pituitary, No Tumor

## Model Architecture

- Input Size: **150 × 150 × 3**
- CNN Layers:
  - Conv2D (32) → MaxPooling
  - Conv2D (64) → MaxPooling
  - Conv2D (128) → MaxPooling
  - Conv2D (128) → MaxPooling
- Fully Connected:
  - Dense (512) + Dropout (0.5)
- Output:
  - Dense (4) + Softmax

## Model Performance

- **Test Accuracy:** **97%**
- Evaluation performed using:
  - Accuracy and loss curves
  - Confusion matrix
  - Classification report

## Web Application

The Streamlit application allows users to:
- Upload a brain MRI image
- Display the predicted tumor class
- Show prediction confidence
- Visualize important regions using Grad-CAM heatmaps

## How to Run the Project

- Install Required Packages:

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn streamlit opencv-python pillow kagglehub pydot graphviz
```

- Run the Streamlit App:

```bash
streamlit run app.py
```

## Note

This project is developed only for academic purposes and should not be used for real medical diagnosis.