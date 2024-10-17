
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define chaotic activation function
def chaotic_neuron(x):
    return tf.sin(x * tf.random.uniform(shape=[], minval=0.5, maxval=1.5))

# Improved Chaotic-CNN model
def build_chaotic_cnn():
    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='chaotic_neuron'))  # Chaotic neuron activation
    model.add(layers.MaxPooling2D((2, 2)))

    # Fully connected layers
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))

    # Chaotic neuron layer
    model.add(layers.Dense(128, activation=chaotic_neuron))
    model.add(layers.Dense(10, activation='softmax'))

    return model

# Build the model
chaotic_cnn = build_chaotic_cnn()

# Compile the model
chaotic_cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = chaotic_cnn.fit(x_train, y_train, epochs=25, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = chaotic_cnn.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc:.2f}")

# Plot training history
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='test accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Classification report
y_pred = np.argmax(chaotic_cnn.predict(x_test), axis=1)
print(classification_report(y_test, y_pred, target_names=[
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']))
