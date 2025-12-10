# ======================================================
# CSA102 - Neural Network on Image Dataset (Fashion MNIST)
# ======================================================

# 1. IMPORT LIBRARIES
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np

print("TensorFlow version:", tf.__version__)

# 2. LOAD DATASET
# Fashion MNIST: 60,000 train + 10,000 test (10 clothing classes)
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

print("Training data shape:", x_train.shape)
print("Test data shape:", x_test.shape)

# 3. PREPROCESSING
# Normalize pixel values from [0, 255] to [0, 1]
x_train = x_train / 255.0
x_test  = x_test  / 255.0

# 4. VISUALIZE SAMPLE IMAGES
plt.figure(figsize=(8, 8))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(class_names[y_train[i]])
    plt.axis('off')
plt.tight_layout()
plt.show()

# 5. BUILD THE NEURAL NETWORK MODEL (ANN)
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),   # 28x28 → 784
    layers.Dense(128, activation='relu'),   # hidden layer
    layers.Dense(10, activation='softmax')  # output layer (10 classes)
])

print("\nMODEL SUMMARY")
model.summary()

# 6. COMPILE THE MODEL
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 7. TRAIN THE MODEL
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(x_test, y_test)
)

# 8. EVALUATE THE MODEL
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_acc)

# 9. PLOT TRAINING vs VALIDATION ACCURACY
plt.figure()
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.show()

# 10. PLOT TRAINING vs VALIDATION LOSS
plt.figure()
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.show()

# 11. SAMPLE PREDICTION
idx = 0  # change this number to test another image
img = x_test[idx]

plt.imshow(img, cmap='gray')
plt.axis('off')

pred = model.predict(np.expand_dims(img, axis=0), verbose=0)
pred_class = np.argmax(pred)

plt.title(f"Predicted: {class_names[pred_class]} | True: {class_names[y_test[idx]]}")
plt.show()
