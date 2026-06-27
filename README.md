# MNIST Digit Classification: Model Comparison

An interactive [Streamlit](https://streamlit.io/) application that explores and compares the performance of three classic machine learning architectures on the MNIST handwritten digit dataset.

## 🧠 Overview
This project provides a visual, hands-on environment to understand how different neural network architectures interpret image data. By training models from scratch, users can observe the trade-offs between model complexity, architectural design, and predictive accuracy.

### Models Compared
* **Perceptron:** A single-layer linear classifier. Excellent for understanding the limitations of linear decision boundaries.
* **ANN (Artificial Neural Network):** A multi-layer perceptron (MLP) utilizing ReLU activations to capture non-linear patterns.
* **CNN (Convolutional Neural Network):** A modern deep learning architecture using convolutional and pooling layers to preserve spatial hierarchies in images.

## 🚀 Key Features
* **Interactive Configuration:** Adjust training epochs and preview data directly in the UI.
* **Performance Metrics:** Real-time visualization of training curves, test accuracy, and confusion matrices.
* **Visual Analysis:** View side-by-side predictions to see where different models succeed or fail.
* **Educational Insights:** Detailed explanations of why CNNs outperform simpler models for vision tasks.

## 🛠 Tech Stack
* **Frontend:** [Streamlit](https://streamlit.io/)
* **Machine Learning:** [TensorFlow/Keras](https://www.tensorflow.org/), [Scikit-learn](https://scikit-learn.org/)
* **Data:** MNIST Dataset

## 💻 How to Run Locally

### Prerequisites
Ensure you have [Python 3.9+](https://www.python.org/) installed.

### Installation
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/RajRover/MNIST-Digit-Classification.git]((https://github.com/RajRover/MNIST-Digit-Classification.git))
   cd mnist-model-comparison
