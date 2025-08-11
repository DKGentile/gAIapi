import tensorflow as tf
import tensorflow_datasets as tfds
import json

#only using PlantVillage dataset rn
dataset, info = tfds.load(
    "plant_village",
    split=["train[:80%]", "train[80%:]"],
    as_supervised=True,
    with_info=True
)

train_ds, test_ds = dataset
num_classes = info.features["label"].num_classes

#normalize and batch data
IMG_SIZE = (128, 128) 

def format_image(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_ds = train_ds.map(format_image).shuffle(1000).batch(32)
test_ds = test_ds.map(format_image).batch(32)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(*IMG_SIZE, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(train_ds, validation_data=test_ds, epochs=5)

#saving class IDs
class_names = info.features["label"].names

#saving model
model.save("../models/plant_classifier.keras")
model.export("../models/plant_classifier")

#exporting class names (for API use)
import json
with open("../models/class_names.json", "w") as f:
    json.dump(class_names, f)

print("Saved Classes: ", class_names)


