# SynapticHub 1.0 🚀

# Table of Contents

| Algorithms                    | Project                                                                                                   | Description                                                                                                 |
| --------------------------   | --------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Convolutional Neural Network | [Handwritten Digit Recognition - MNIST](https://github.com/Shashwat-Akhilesh-Shukla/Synaptic-Hub-1.0/blob/main/MNIST_DIGIT_RECOGNITION.ipynb)             | Implementation of a Convolutional Neural Network (CNN) for handwritten digit recognition using the MNIST dataset.                                      |
|  | [Binary Classifier](https://github.com/Shashwat-Akhilesh-Shukla/Synaptic-Hub-1.0/blob/main/cnn_binary_classifier.py)                       | Development of a binary classifier with CNN architecture for class differentiation.                                      |
| | [Fashion MNIST Classification](https://github.com/Shashwat-Akhilesh-Shukla/Synaptic-Hub-1.0/blob/main/fashion_mnist_cnn_classification.py) | CNN model training for fashion-related object classification using the Fashion MNIST dataset.                                            |
|  | [CIFAR10 Classification](https://github.com/Shashwat-Akhilesh-Shukla/Synaptic-Hub-1.0/blob/main/cifar10_cnn_classification.py)                   | CNN utilization for object detection in the CIFAR10 dataset with 10 different classes.                                               |
|  | [CIFAR100 Classification](https://github.com/Shashwat-Akhilesh-Shukla/Synaptic-Hub-1.0/blob/main/cnn_cifar100.py)                                                       | CNN implementation for object detection in the CIFAR100 dataset containing 100 distinct classes.                                        |
|  | [Facial Expression Recognition](https://github.com/Shashwat-Akhilesh-Shukla/Synaptic-Hub-1.0/blob/main/facial_expression_recognition.py)            | Facial expression recognition system creation using CNN architecture for emotion detection from images.                                       |
| Long Short-Term Memory       | [IMDB Sentiment Analysis](https://github.com/Shashwat-Akhilesh-Shukla/Synaptic-Hub-1.0/blob/main/lstm_imdb_sentiment_analysis.py)               | Long Short-Term Memory (LSTM) application for sentiment analysis on the IMDB dataset.                           |
| Gated Recurrent Neural Networks | [GRU Sentiment Analysis](https://github.com/Shashwat-Akhilesh-Shukla/Synaptic-Hub-1.0/blob/main/imdb_GRU.ipynb) | Sentiment analysis on the IMDB dataset using Gated Recurrent Neural Networks (GRU).|
| Simple Neural Network | [Text Generation](https://github.com/Shashwat-Akhilesh-Shukla/Synaptic-Hub-1.0/blob/main/RNN_Text_Generation.ipynb) | Text generation with recurrent neural networks.|











## Installation

### Tensorflow 2.0 and Keras

Ensure you have Tensorflow 2.0 and Keras installed:

```bash
pip install tensorflow==2.0
pip install keras
```

### Google Colab Setup (Preferred Alternative)

For enhanced flexibility, use Google Colab:

1. Open the notebook in Colab.
2. Connect to a hosted runtime.
3. Execute the notebook cells.

## Common Steps in a Deep Learning Project

1. **Importing the necessary libraries:**
   - import libraries like Tensorflow and Keras for building neural nets.

2. **Data Preprocessing:**
   - preprocess the data to get it into a format which is suitable for training neural nets.

3. **Build the Model:**
   - use Tensorflow and Keras to build the model. add the required number of layers and required activation functions.

4. **Compile the model:**
   - compile the neural network to get it ready for training.

5. **Train your model:**
   - Train your model on the dataset.

6. **Evaluate the model:**
   - evaluate the model on test set and validation set and compare the accuracies on Train and Validation set.

## Libraries Involved

- **Tensorflow and Keras:** For constructing and training deep learning models.
- **NumPy:** Essential for numerical operations.
- **Matplotlib and Seaborn:** Tools for data visualization.

## Contact Information

For questions or collaborations:

- **Email:** shashwatakhileshshukla@gmail.com
- **LinkedIn:** [Shashwat Shukla](https://www.linkedin.com/in/shashwat-shukla-2a90a525b/)

## FAQs

### What is Deep Learning?

Deep learning, a subset of machine learning, employs neural networks to emulate human-like decision-making.

### What is Tensorflow?

Tensorflow, an open-source machine learning framework by Google, is used for building and training deep learning models.

### What are Hyperparameters?

Hyperparameters are configuration settings that control the learning process of a neural network, e.g., learning rate and batch size.

### What are Tensors?

Tensors are multi-dimensional arrays representing data in deep learning, ranging from scalars to matrices.

### What are Activation Functions?

Activation functions introduce non-linearities to neural networks. Common functions include:
- **ReLU (Rectified Linear Unit):** `max(0, x)`
- **Sigmoid:** `1 / (1 + exp(-x))`
- **TanH:** `(exp(x) - exp(-x)) / (exp(x) + exp(-x))`

### What is Transfer Learning?

Transfer learning is a technique where a pre-trained model's knowledge is applied to a new, similar task, often saving time and resources.

### What is Overfitting?

Overfitting occurs when a model learns the training data too well, losing its ability to generalize to new, unseen data.

### What is a Convolutional Neural Network (CNN)?

A CNN is a type of neural network designed for image recognition, utilizing convolutional layers to capture spatial hierarchies.

Feel free to explore the fascinating world of deep learning! 🤖📈
