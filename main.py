import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import pathlib

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


'''
Download the dataset
'''
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)


'''
Exploring the data with count and showing certain instances of it
'''
def exploring_dataset(data_dir):
    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(image_count)

    roses = list(data_dir.glob('roses/*'))
    image_rose_1 = PIL.Image.open(str(roses[0]))
    image_rose_2 = PIL.Image.open(str(roses[1]))
    image_rose_1.show()
    image_rose_2.show()

    tulips = list(data_dir.glob('tulips/*'))
    image_tulip_1 = PIL.Image.open(str(tulips[0]))
    image_tulip_2 = PIL.Image.open(str(tulips[1]))
    image_tulip_1.show()
    image_tulip_2.show()

# exploring_dataset(data_dir=data_dir)


# Preprocessing with Keras
batch_size = 32
img_height = 180
img_widht = 180

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_widht),
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_widht),
    batch_size=batch_size
)

class_names = train_ds.class_names
print("Class names:", class_names)


'''
Visualization of the data currently working on. It is importnat the data is an 
object of the tf.keras.preprocessing.image_dataset_from_directory() since the code
is relaying on this entirly.
'''
def visualize_data(data):
    plt.figure(figsize=(10, 10))
    for images, labels in data.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.show()

# visualize_data(data=train_ds)

# PRINT TEST - looking at tensor shapes of images and labels
for image_batch, label_batch in train_ds:
    print(image_batch.shape)         # 32 images, 180x180, in RGB and therefore the 3 at the end. (32, 180, 180, 3)
    print(label_batch.shape)         # (32,) corresponding to the images
    break
    
'''
CONFIUGRE THE DATASET PERFORMANCE
'''
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

'''
STANDARDIZE THE DATA
'''
# standarlize the data from 0 - 255 to 0 - 1
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
#                      pass x and y into normalization_layer
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, label_batch = next(iter(normalized_ds))

#PRINT TEST
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))

'''
DATA AUGMENTATION
'''
data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height, img_widht, 3)),
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.RandomZoom(0.1),
    ]
)

'''
VISAULIZE OF DATA AUGMENTATION 
'''
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
  for i in range(9):
    augmented_images = data_augmentation(images)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[0].numpy().astype("uint8"))
    plt.axis("off")
plt.show()



'''
CREAT THE MODEL
'''
num_classes = 5 
if num_classes != len(class_names):
    print('ERROR ERROR LABELS ARE DIFFERENT')

model = Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_widht, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

'''
COMPILE THE MODEL
'''
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

'''
MODEL SUMMARY
'''
model.summary()

'''
TRAIN THE MODEL
'''
epochs=15
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

'''
VISUALIZE TRANING RESULTS
'''
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

'''
PREDICTON ON NEW DATA
'''
sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

img = keras.preprocessing.image.load_img( 
    sunflower_path, target_size=(img_height, img_widht)
)

img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # creating a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)



