
from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_cnn.keras"
UPLOAD_DIR = BASE_DIR / "static" / "uploads"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(MODEL_PATH)

print("CNN model loaded successfully.")
print("Model output classes:", model.output_shape[-1])


def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def predict_mri(image_path):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize((128, 128))
        image_array = np.array(image, dtype=np.float32)

    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)[0]
    predicted_index = int(np.argmax(predictions))

    return (
        CLASS_NAMES[predicted_index],
        float(predictions[predicted_index]) * 100
    )


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_url = None
    error = None

    if request.method == "POST":
        file = request.files.get("mri_image")

        if not file or not file.filename:
            error = "Please select an MRI image."

        elif not allowed_file(file.filename):
            error = "Upload a PNG, JPG, or JPEG image."

        else:
            safe_name = secure_filename(file.filename)

            if not safe_name:
                error = "Invalid filename."

            else:
                try:
                    # Validate the uploaded image before saving.
                    file.stream.seek(0)
                    with Image.open(file.stream) as image:
                        image.verify()

                    file.stream.seek(0)
                    image_path = UPLOAD_DIR / safe_name
                    file.save(image_path)

                    prediction, confidence = predict_mri(image_path)
                    image_url = f"uploads/{safe_name}"

                except (UnidentifiedImageError, OSError, ValueError):
                    error = "The uploaded file is not a valid image."

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_url=image_url,
        error=error
    )


@app.errorhandler(413)
def file_too_large(error):
    return "Image too large. Please upload an image under 8 MB.", 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
