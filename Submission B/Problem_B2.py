# =============================================================================
# PROBLEM B2
#
# Build a classifier for the Fashion MNIST dataset.
# The test will expect it to classify 10 classes.
# The input shape should be 28x28 monochrome. Do not resize the data.
# Your input layer should accept (28, 28) as the input shape.
#
# Don't use lambda layers in your model.
#
# Desired accuracy AND validation_accuracy > 83%
# =============================================================================

import tensorflow as tf
import urllib.request
import numpy as np

def solution_B2():
    # Mengimpor dataset Fashion MNIST dari TensorFlow
    fashion_mnist = tf.keras.datasets.fashion_mnist

    # Mengunduh dan memuat data, kemudian menormalkan gambar dengan membagi nilai piksel dengan 255.0
    (training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()
    training_images=training_images.reshape(60000, 28, 28, 1)
    training_images=training_images / 255.0
    test_images = test_images.reshape(10000, 28, 28, 1)
    test_images=test_images/255.0

    # Mendefinisikan model dengan beberapa lapisan Conv2D dan MaxPooling2D, diakhiri dengan lapisan Dense dengan 10 neuron dan fungsi aktivasi softmax
    model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPooling2D(2, 2),
      tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
      tf.keras.layers.MaxPooling2D(2, 2),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(256, activation='relu'),
      tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Mengkompilasi model dengan optimizer 'adam' dan loss 'sparse_categorical_crossentropy'
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    # Melatih model dengan data pelatihan dan validasi
    history = model.fit(training_images, training_labels, validation_data=(test_images, test_labels), epochs=20, verbose=1)

    # Mencetak akurasi dan akurasi validasi
    print("Desired accuracy: ", history.history['accuracy'][-1])
    print("Validation accuracy: ", history.history['val_accuracy'][-1])

    return model

# Menyimpan model sebagai file .h5
if __name__ == '__main__':
    model = solution_B2()
    model.save("model_B2.h5")