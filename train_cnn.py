```python
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models

# Dataset paths
BASE_DIR = Path("brisc2025/classification_task")
TRAIN_DIR = BASE_DIR / "train"
TEST_DIR = BASE_DIR / "test"

# Settings
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42

# Check dataset folders
if not TRAIN_DIR.exists():
    raise FileNotFoundError(f"Training folder not found: {TRAIN_DIR}")

if not TEST_DIR.exists():
    raise FileNotFoundError(f"Test folder not found: {TEST_DIR}")

# Load training images and create validation split
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Load separate test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
print("Classes:", class_names)

# Memory-friendly data loading: no in-memory cache
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.shuffle(
    1000, seed=SEED
).prefetch(1)

val_ds = val_ds.prefetch(1)
test_ds = test_ds.prefetch(1)

# Build CNN
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Rescaling(1.0 / 255),

    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Save the best model
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "best_cnn.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
]

# Train the CNN
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=callbacks
)

# Evaluate the best saved model on the held-out test set
best_model = tf.keras.models.load_model("best_cnn.keras")

test_loss, test_accuracy = best_model.evaluate(test_ds)

print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.2%}")

print("\nTraining and evaluation complete.")
print("Model saved as best_cnn.keras")
```
