import wandb
import os
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

os.environ['WAND_NOTEBOOK_NAME'] = 'Q1'

wandb.login()



# Initialize Weights & Biases
wandb.init(project='Assignment 1', entity='cs22m070', name='question_1')

# Define class names
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Load Fashion-MNIST dataset
(train_images, train_labels), (_, _) = fashion_mnist.load_data()

# Collect one sample image for each class
sample_images = []
sample_labels = []

for i in range(len(classes)):
    for j in range(len(train_labels)):
        if train_labels[j] == i:
            sample_images.append(train_images[j])
            sample_labels.append(classes[i])
            break

# Display sample images
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for ax, image, label in zip(axes.flat, sample_images, sample_labels):
    ax.imshow(image, cmap='binary')
    ax.set_title(label)
    ax.axis('off')

# Log the plot
wandb.log({"Sample Images": fig})
wandb.save()
wandb.finish()
