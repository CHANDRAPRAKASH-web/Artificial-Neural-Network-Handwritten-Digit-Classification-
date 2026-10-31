# Handwritten Digit Classification Using ANN

## 📌 Overview
This project implements an Artificial Neural Network (ANN) using TensorFlow and Keras to classify handwritten digits from 0 to 9 using the MNIST dataset. The model learns patterns from grayscale images and predicts the corresponding digit.

## 🎯 Objectives
- Build and train an ANN using TensorFlow and Keras.
- Classify handwritten digits using the MNIST dataset.
- Evaluate model performance using loss and accuracy.
- Visualize training and validation performance using learning curves.

## 📊 Dataset
The project uses the **MNIST handwritten digit dataset**, which contains 70,000 grayscale images.

- **Training images:** 60,000
- **Testing images:** 10,000
- **Image dimensions:** 28 × 28 pixels
- **Number of classes:** 10 (digits 0–9)

Pixel values are normalized from the range 0–255 to 0–1.

## 🧠 Model Architecture
The ANN consists of the following layers:

| Layer | Configuration | Purpose |
|---|---|---|
| Flatten | 28 × 28 → 784 features | Converts images into one-dimensional vectors |
| Dense | 64 neurons, ReLU | Learns patterns from input features |
| Dense | 10 neurons, Softmax | Predicts probabilities for digits 0–9 |

**Total trainable parameters:** 50,890

## ⚙️ Technologies Used
- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- MNIST Dataset

## 🔧 Training Configuration
- **Optimizer:** Adam
- **Loss Function:** Sparse Categorical Cross-Entropy
- **Evaluation Metric:** Accuracy
- **Epochs:** 10
- **Batch Size:** 32
- **Validation Split:** 20%

## 📈 Model Evaluation
The model is evaluated using the separate MNIST test dataset.

The project reports:
- Test Loss
- Test Accuracy
- Training Loss Curve
- Validation Loss Curve
- Training Accuracy Curve
- Validation Accuracy Curve

The learning curves help analyze the model's learning progress and identify potential overfitting.

## 🚀 Installation and Execution

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### 2. Install Dependencies
```bash
pip install tensorflow numpy matplotlib
```

### 3. Run the Program
Save the Python code as `ann_mnist.py` and execute:

```bash
python ann_mnist.py
```

The program trains the ANN, displays test loss and accuracy, and plots the training and validation curves.

## 📁 Project Structure
```text
ANN-Handwritten-Digit-Classification/
├── ann_mnist.py
└── README.md
```

## ✅ Expected Outcome
The trained ANN should classify handwritten digits with good accuracy. The actual test accuracy and loss depend on the training environment and execution.

## 🔮 Future Improvements
- Add more hidden layers.
- Experiment with different numbers of neurons.
- Implement dropout to reduce overfitting.
- Compare ANN performance with a Convolutional Neural Network (CNN).
- Add a user interface for predicting digits from uploaded images.

## 👨‍💻 Author
Developed as a deep learning experiment in handwritten digit classification using TensorFlow and Keras.