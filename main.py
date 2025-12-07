import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras import Sequential, backend
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report

# 1. baca dataset di lokal
BASE_PATH = ""
train_dir = BASE_PATH + "Training"
test_dir  = BASE_PATH + "Testing"

# 2. normalisasi warna
def normalize_color(image_dir):
    for subdir in os.listdir(image_dir):
        subdir_path = os.path.join(image_dir, subdir)
        for filename in os.listdir(subdir_path):
            fpath = os.path.join(subdir_path, filename)
            try:
                img = Image.open(fpath)
                img = img.convert("RGB")
                img.save(fpath)
            except:
                pass

normalize_color(train_dir)
normalize_color(test_dir)

# 3. normalisasi ukuran gambar
DIMENSIONS = (200, 200)

# 4. augmentasi dan data generator
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=50,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=(0.8,1.2),
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=DIMENSIONS,
    class_mode='categorical',
    batch_size=32
)

test_generator = valid_datagen.flow_from_directory(
    test_dir,
    target_size=DIMENSIONS,
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)

# 5. build model mobilenetV2
def build_model():
    backend.clear_session()
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(200,200,3))
    base.trainable = False

    model = Sequential([
        base,
        GlobalAveragePooling2D(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(4, activation="softmax")
    ])
    
    model.summary()
    return model

model = build_model()

# 6. callback
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ModelCheckpoint("best_mobilenetv2.keras", monitor="val_accuracy", save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5)
]

# 7. training model
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=50,
    callbacks=callbacks
)

# 8. plot akurasi dan loss 
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Loss")
plt.legend(["train","val"])

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")
plt.legend(["train","val"])
plt.show()

# 9. classification report
y_prob = model.predict(test_generator)
y_pred = np.argmax(y_prob, axis=1)
y_true = test_generator.classes
labels = list(test_generator.class_indices.keys())

print("\n=== Classification Report ===\n")
print(classification_report(y_true, y_pred, target_names=labels))

# 10. test gambar
def predict_image(path):
    img = Image.open(path).convert("RGB")
    plt.imshow(img)
    plt.axis("off")
    plt.show()

    img = img.resize(DIMENSIONS)
    img = np.array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    class_id = np.argmax(pred)
    print("Prediction:", labels[class_id])
    print("Probabilities:", pred)

# 11. Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix - MobileNetV2")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

