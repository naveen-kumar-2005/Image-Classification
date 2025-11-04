# Convolutional Deep Neural Network for Image Classification

## AIM

To develop a Convolutional Deep Neural Network (CNN) model for image classification and to verify the response for new images.

---
## Problem Overview and Dataset

The objective of this project is to create a CNN that can categorize images of fashion items from the Fashion MNIST dataset. This dataset includes grayscale images of clothing and accessories such as T-shirts, trousers, dresses, and footwear. The task is to accurately predict the correct category for each image while ensuring the model is efficient and robust.

* **Training data:** 60,000 images
* **Test data:** 10,000 images
* **Classes:** 10 fashion categories

The CNN consists of multiple convolutional layers with activation functions, followed by pooling layers, and ends with fully connected layers to output predictions for all 10 categories.

## NEURAL NETWORK MODEL

### CNN Model Architecture

![alt text](/Images/image-4.png)
---

## Design Steps:

### Step 1: Define the Objective

Formulate the task of classifying fashion items (shirts, shoes, bags, etc.) using a CNN model.

### Step 2: Dataset Preparation

Load the Fashion MNIST dataset and split it into training and testing sets.

### Step 3: Data Preprocessing

* Convert images to tensors
* Normalize pixel intensity values
* Use DataLoaders for batching and shuffling

### Step 4: Construct the CNN

Design a neural network with:

* Convolutional layers to extract features
* ReLU activations for non-linearity
* Pooling layers to reduce spatial dimensions
* Fully connected layers for final classification

### Step 5: Train the Model

* Use CrossEntropyLoss as the loss function
* Optimize with the Adam optimizer
* Train over multiple epochs, monitoring loss and accuracy

### Step 6: Evaluate Performance

* Test the trained model on unseen images
* Compute accuracy, precision, recall, and F1-score
* Generate a confusion matrix to analyze misclassifications

### Step 7: Deployment and Visualization

* Save the trained model for future use
* Visualize sample predictions
* Integrate the model into applications if required

---

## PROGRAM

### Name: Naveen Kumar.R

### Register Number: 212223230139

```python
import torch
import torch.nn as nn
import torch.optim as optim

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 3 * 3)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize Model, Loss Function, Optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Function
def train_model(model, train_loader, num_epochs=3):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print("Name: Naveen Kumar.R")
        print("Register Number: 212223230139")
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
```
---

## OUTPUT

### Training Loss per Epoch

![alt text](/Images/image.png)

### Confusion Matrix

![alt text](/Images/image-2.png)

### Classification Report

![alt text](/Images/image-1.png)

### New Sample Data Prediction

![alt text](/Images/image-3.png)

---

## RESULT

Thus, we successfully developed a **Convolutional Deep Neural Network (CNN)** for image classification using the Fashion MNIST dataset. 
