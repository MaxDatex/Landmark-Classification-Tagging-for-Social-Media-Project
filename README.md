# Landmark Classification Project

This project focuses on building a landmark classification system using Convolutional Neural Networks (CNNs). The system is designed to identify landmarks in images, even in the absence of location metadata.

## Project Structure

The project is organized as follows:

```
└── ./
    ├── src
    │   ├── __init__.py
    │   ├── create_submit_pkg.py
    │   ├── data.py
    │   ├── helpers.py
    │   ├── model.py
    │   ├── optimization.py
    │   ├── predictor.py
    │   ├── train.py
    │   └── transfer.py
    ├── app.html
    ├── app.ipynb
    ├── cnn_from_scratch.html
    ├── cnn_from_scratch.ipynb
    ├── requirements.txt
    ├── transfer_learning.html
    └── transfer_learning.ipynb
```

###   Key Components

* **`app.ipynb`**:   This file contains the code for a simple application that uses a trained model to classify landmark images. It includes functionality for uploading an image, displaying it, and showing the top predicted landmark classes with their probabilities.
* **`cnn_from_scratch.ipynb`**:   This Jupyter Notebook contains the code to train a CNN model from scratch. It covers data loading and visualization, model definition, training loop, and testing. It also includes steps to export the trained model using Torch Script.
* **`transfer_learning.ipynb`**:   This Jupyter Notebook focuses on using transfer learning to train a landmark classifier. It loads a pre-trained model, modifies it for the classification task, trains it, and evaluates its performance. Like `cnn_from_scratch.ipynb`, it also exports the trained model.
* **`src/`**:   This directory contains the core Python modules for the project.
    * `data.py`:   Provides functionality for loading and preprocessing the image data. It includes data loaders for training, validation, and testing, as well as data visualization tools.
    * `helpers.py`:   Contains helper functions for tasks such as downloading and extracting the dataset, computing mean and standard deviation for normalization, and setting up the environment.
    * `model.py`:   Defines the CNN architecture (`MyModel`) used for landmark classification.
    * `optimization.py`:   Implements functions for getting the loss function (CrossEntropyLoss) and optimizer (SGD or Adam).
    * `predictor.py`:   Contains the `Predictor` class, which wraps the trained model for inference, including preprocessing transforms.
    * `train.py`:   Provides functions for training and validating the model, as well as for testing its performance.
    * `transfer.py`:   Implements transfer learning functionality, allowing the use of pre-trained models for the classification task.

###   Key Files and their Functionalities

* **`app.ipynb`**:   This script implements a user interface for landmark classification.  It allows users to upload an image, and then displays the image along with the top 5 predicted landmark names and their associated probabilities, using a pre-trained model loaded from "checkpoints/transfer\_exported.pt".
* **`transfer_learning.ipynb`**:   This notebook demonstrates transfer learning. It defines hyperparameters and then loads a pre-trained model using `get_model_transfer_learning` (specifically "resnext101\_32x8d"). It then trains this model, evaluates it, and exports it using Torch Script. Finally, it loads the exported model and evaluates its performance.
* **`src/data.py`**:   This module is crucial for data handling. It provides the `get_data_loaders` function, which creates PyTorch DataLoaders for train, validation, and test datasets.. These data loaders handle batching, shuffling, and parallel loading of the data. The module also includes functions for visualizing data batches and applies necessary transformations to the images, such as resizing, cropping, normalization, and data augmentation.
* **`src/helpers.py`**:   This module contains utility functions that support the main functionalities of the project. It includes `setup_env` for setting up the environment, `get_data_location` to determine the dataset's location and functions to download and extract the dataset. It also provides functions to compute the mean and standard deviation of the dataset for normalization.
* **`src/model.py`**:   This file defines the `MyModel` class, which is a PyTorch `nn.Module` that implements the CNN architecture.
* **`src/optimization.py`**:   This module focuses on providing the necessary tools for optimizing the model during training. It includes the `get_loss` function, which returns an instance of the `CrossEntropyLoss`, and the `get_optimizer` function, which provides instances of optimization algorithms such as SGD or Adam.
* **`src/predictor.py`**:   The `predictor.py` script defines the `Predictor` class, which encapsulates the model and the necessary transformations for inference..
* **`src/train.py`**:   This module contains the core logic for training and validating the CNN model. The `train_one_epoch` function performs one epoch of training, the `valid_one_epoch` function evaluates the model on the validation set, and the `optimize` function orchestrates the entire training process. Additionally, the `one_epoch_test` function is provided to evaluate the trained model on the test dataset.
* **`src/transfer.py`**:   This script provides the `get_model_transfer_learning` function, which facilitates transfer learning by loading a pre-trained model, freezing its parameters, and replacing the final fully connected layer.

###   Implementation Details

In this project, I implemented a landmark classification system through the following key steps:

1.  **Application Development:**
    * I created a user-friendly application using `app.ipynb` that allows users to upload landmark images and receive predictions. The application loads a pre-trained model and displays the top 5 predicted landmarks with their probabilities.
2.  **Training a CNN from Scratch:**
    * I developed a CNN model from scratch using the code in the `cnn_from_scratch.ipynb` notebook. This involved loading and preprocessing the data, defining the model architecture, and training the model.
    * I carefully selected hyperparameters and used visualization techniques to understand the training process. I also exported the trained model using Torch Script.
3.  **Transfer Learning Implementation:**
    * I implemented transfer learning using the `transfer_learning.ipynb` notebook. I loaded a pre-trained model, adapted it for the landmark classification task, and trained it efficiently.
    * I compared the performance of the transfer learning approach with the model trained from scratch.
4.  **Data Handling:**
    * I used the `src/data.py` module to manage the image data. This involved creating PyTorch DataLoaders for efficient batching and loading of training, validation, and test sets.
    * I applied various transformations to the images, including resizing, cropping, normalization, and data augmentation, to prepare them for model training and to improve the model's generalization.
5.  **Model Architecture:**
    * I designed a CNN architecture (`MyModel`) in `src/model.py` using convolutional layers, batch normalization, ReLU activations, max pooling, and fully connected layers with dropout to classify landmark images.
6.  **Training and Optimization:**
    * I implemented training and validation procedures in `src/train.py` to train the CNN model. This involved defining functions for training over a single epoch, validating the model, and an overarching `optimize` function to manage the training loop.
    * I utilized `src/optimization.py` to handle the loss function and optimizer. I used CrossEntropyLoss for the classification task and implemented the ability to use either SGD or Adam as the optimizer.
7.  **Inference:**
    * I created a `Predictor` class in `src/predictor.py` to encapsulate the trained model and the necessary preprocessing steps for making predictions on new images.

This implementation allowed me to effectively build and evaluate a landmark classification system, incorporating training from scratch, transfer learning, and application development. 
