# 🚦 Traffic Sign Recognition using Deep Learning

A deep learning project for recognizing and classifying German traffic signs using the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset.

The project implements and compares three different deep learning approaches:

- Custom CNN
- MobileNetV3-Small
- ResNet18

After training and evaluation, **ResNet18 achieved the best overall performance** and was selected as the final model.

---

## 🎯 Project Goal

The goal of this project is to build a computer vision system that receives an image of a traffic sign and predicts its corresponding traffic sign class.

The complete workflow includes:

```text
Dataset Selection
       ↓
Data Inspection & Cleaning
       ↓
Image Resizing
       ↓
Normalization
       ↓
Data Augmentation
       ↓
Train / Validation / Test Split
       ↓
Baseline CNN
       ↓
Transfer Learning
       ↓
Model Training
       ↓
Evaluation
       ↓
Model Comparison
       ↓
Error Analysis
       ↓
Final Model Selection
       ↓
Streamlit Testing Application
```

---

## 📊 Dataset

The project uses the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset.

The dataset contains **43 different traffic sign classes**, including:

- Speed limits
- Stop
- Yield
- No entry
- No passing
- Road work
- Traffic signals
- Pedestrians
- Keep right
- Keep left
- Roundabout mandatory
- Dangerous curves

The dataset is used as a multi-class image classification problem.

---

## 🧹 Data Inspection and Cleaning

Before training, the dataset was inspected and prepared for the training pipeline.

The preprocessing stage includes:

- Checking image readability
- Detecting invalid or unusable images
- Verifying class labels
- Organizing images according to their classes
- Removing invalid samples when necessary

This ensures that corrupted or unusable images do not negatively affect the training process.

---

## 📐 Image Preprocessing

All input images are resized to a consistent input resolution before being passed to the neural networks.

Normalization is then applied to create more stable inputs for the models, particularly for the transfer-learning models.

The preprocessing pipeline is implemented using **PyTorch** and **Torchvision**.

---

## 🔄 Data Augmentation

Data augmentation is applied to the training images to increase data diversity and reduce overfitting.

The augmentation pipeline includes techniques such as:

- Rotation
- Cropping
- Brightness changes
- Horizontal flipping where appropriate

Horizontal flipping is handled carefully because the direction of some traffic signs is semantically important.

For example, flipping a left-direction sign horizontally can create an image corresponding to a right-direction sign, which could introduce incorrect training labels.

---

## 📚 Dataset Split

The dataset is divided into three subsets:

| Split | Purpose |
|---|---|
| Training | Used to train the neural networks |
| Validation | Used to monitor training and select the best checkpoint |
| Test | Used for final evaluation on unseen samples |

The test set is not used for model training.

---

# 🧠 Models

Three different architectures were trained and evaluated.

## 1. Custom CNN

A custom Convolutional Neural Network was implemented as the baseline model.

CNNs are particularly suitable for image classification because they can learn visual features such as:

- Edges
- Shapes
- Patterns
- Textures

The CNN provides a baseline for comparing the performance of the more advanced architectures.

---

## 2. MobileNetV3-Small

**MobileNetV3-Small** was evaluated using transfer learning.

MobileNetV3 is designed to provide strong image classification performance while keeping the architecture relatively lightweight.

This makes it a useful candidate for applications where computational efficiency is important.

---

## 3. ResNet18

**ResNet18** was also evaluated using transfer learning.

ResNet uses residual connections that help neural networks learn deeper representations effectively.

Among the three tested architectures, ResNet18 achieved the best overall results and was selected as the final model.

---

# 📈 Model Comparison

The three models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score

The final comparison was:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| CNN | 99.88% | 99.92% | 99.89% | 99.90% |
| MobileNetV3-Small | 99.95% | 99.97% | 99.96% | 99.96% |
| **ResNet18** | **99.97%** | **99.98%** | **99.96%** | **99.97%** |

### 🏆 Best Model

Based on the evaluation results, **ResNet18** was selected as the final model.

Its final test performance was:

```text
Accuracy  : 99.97%
Precision : 99.98%
Recall    : 99.96%
F1-score  : 99.97%
```

The trained model is stored at:

```text
runs/gtsrb_resnet18/best_model.pth
```

---

# 🔍 Confusion Matrix

A confusion matrix was generated to analyze the classification performance of the final model across all 43 classes.

The majority of predictions are concentrated along the main diagonal, indicating that most test samples were correctly classified.

Both the standard and normalized confusion matrices are saved in the corresponding model output directory.

```text
runs/gtsrb_resnet18/
├── confusion_matrix.png
└── confusion_matrix_normalized.png
```

---

# ❌ Error Analysis

The final ResNet18 model was evaluated on **5,882 test images** and achieved an accuracy of approximately **99.97%**.

Only **two test samples were incorrectly classified**.

The observed errors were:

| True Class | Predicted Class | Number of Errors |
|---|---|---:|
| Speed limit (100 km/h) | Speed limit (120 km/h) | 1 |
| No vehicles | No passing for vehicles over 3.5 t | 1 |

These errors occurred between visually similar traffic-sign classes.

For example, the 100 km/h and 120 km/h speed-limit signs have very similar visual structures, making this type of error understandable even for a highly accurate classifier.

The `No vehicles` and `No passing for vehicles over 3.5 t` classes can also have visually similar characteristics.

Overall, the extremely small number of errors indicates that the final model generalizes very well to the test set.

---

# 📊 Training Outputs

Each trained model stores its results under the `runs/` directory.

Example:

```text
runs/
├── gtsrb_cnn/
├── gtsrb_mobilenet_full/
└── gtsrb_resnet18/
```

Each model directory may contain:

```text
best_model.pth
metrics.json
classification_report.json
config.json
history.csv
accuracy_curve.png
loss_curve.png
confusion_matrix.png
confusion_matrix_normalized.png
misclassified_examples.csv
misclassified_examples.png
```

These files allow the training process and evaluation results to be inspected after training.

---

# 🖥️ Streamlit Application

A Streamlit application was created to test the final ResNet18 model interactively.

The application allows the user to:

1. Upload a traffic-sign image.
2. Preprocess the image.
3. Run inference using ResNet18.
4. Display the predicted class.
5. Display the Persian translation of the predicted class.
6. Display the model confidence.
7. Display the top three predictions.

The application uses the trained model:

```text
runs/gtsrb_resnet18/best_model.pth
```

---

## ▶️ Running the Application

First install the required dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python -m streamlit run streamlit_app.py
```

Alternatively, on Windows:

```text
run_app.bat
```

can be used to start the application.

---

# 📁 Project Structure

```text
traffic_sign_ai/
│
├── train.py
├── predict.py
├── compare_results.py
├── inspect_dataset.py
├── streamlit_app.py
├── run_app.bat
│
├── requirements.txt
├── LICENSE
├── README.md
├── README_FA.md
│
├── traffic_ai/
│   ├── __init__.py
│   ├── data.py
│   ├── models.py
│   └── vision.py
│
└── runs/
    ├── model_comparison.csv
    │
    ├── gtsrb_cnn/
    │
    ├── gtsrb_mobilenet_full/
    │
    └── gtsrb_resnet18/
```

---

# 🛠️ Technologies

The project was developed using:

- **Python**
- **PyTorch**
- **Torchvision**
- **NumPy**
- **Pillow**
- **Scikit-learn**
- **Matplotlib**
- **Streamlit**

---

# 🚀 Reproducibility

To reproduce the project:

```bash
git clone <repository-url>
cd traffic_sign_ai
```

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The training script can then be used to train the selected model and generate evaluation outputs.

---

# 📌 Conclusion

This project demonstrates a complete deep-learning workflow for German traffic-sign recognition.

The workflow covered:

- Dataset selection
- Data inspection and cleaning
- Image preprocessing
- Normalization
- Data augmentation
- Train/validation/test splitting
- Baseline CNN development
- Transfer learning
- Multiple model training
- Model evaluation
- Confusion-matrix analysis
- Error analysis
- Model comparison
- Interactive Streamlit testing

Three architectures were evaluated, and **ResNet18 achieved the best overall performance with 99.97% test accuracy and a 99.97% F1-score**.

Therefore, ResNet18 was selected as the final model for the project.