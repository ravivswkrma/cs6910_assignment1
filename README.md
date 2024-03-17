# cs6910_assignment1
Link to wandb report:- https://wandb.ai/cs22m070/Assignment%201/reports/CS6910-Assignment-1-Report--Vmlldzo3MTU2NDg0

Initially, we utilized the `train.py` script to conduct a hyperparameter sweep, incorporating certain sections of the code that were commented out. Towards the end of `train.py`, code for plotting the confusion matrix is also provided but commented out. The `sweep.yaml` file contains the configuration for initializing the sweep with the necessary hyperparameters. Additionally, the file named `Sample Images.png` showcases one sample image from each of the 10 classes in the fashion-mnist dataset. The best set of hyperparameters discovered for the fashion-mnist dataset are as follows:

- Activation function: Rectified Linear Unit (ReLU)
- Batch size: 32
- Learning rate (η): 0.001
- L2 regularization (alpha): 0
- Epochs: 10
- Hidden layer size: 128
- Number of layers: 5
- Optimizer: Adam
- Weight initialization: Xavier
- Loss function: Cross Entropy

This paraphrased version maintains the key details and structure of the original text.

Best Results -

Validation Accuracy - 0.8822 Test Accuracy - 0.8687 Test error - 0.3771
