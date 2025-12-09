import os
import io
import base64

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

from flask import Flask, request, jsonify
from flask_cors import CORS

import tf_keras as keras
import gdown


# ===================== CONFIG =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = "cnn50_100_16_4.h5"
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)

# Set trên Render: MODEL_FILE_ID = "xxxxxxxxxxxxxxxxxxxx"
MODEL_FILE_ID = os.environ.get("MODEL_FILE_ID", "").strip()

CLASS_LABELS = [
    "Banh beo", "Banh chung", "Banh cuon", "Banh mi",
    "Banh trang nuong", "Banh xeo", "Bun dau mam tom",
    "Ca kho to", "Pho", "Xoi xeo"
]

CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", 98))


# ===================== APP =====================
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


# ===================== MODEL DOWNLOAD =====================
def ensure_model_file():
    """Nếu model chưa tồn tại ở server thì tải từ Google Drive bằng FILE_ID."""
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
        print(f"[OK] Model exists: {MODEL_PATH} ({os.path.getsize(MODEL_PATH)/1024/1024:.2f} MB)")
        return True

    if not MODEL_FILE_ID:
        print("[ERR] Missing MODEL_FILE_ID env. Set it in Render Environment.")
        return False

    url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
    print(f"[INFO] Downloading model from Google Drive: {url}")
    try:
        gdown.download(url, MODEL_PATH, quiet=False)
    except Exception as e:
        print(f"[ERR] Download failed: {e}")
        return False

    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
        print(f"[OK] Downloaded model to: {MODEL_PATH} ({os.path.getsize(MODEL_PATH)/1024/1024:.2f} MB)")
        return True

    print("[ERR] Model file not found after download.")
    return False


# ===================== LOAD MODEL =====================
model = None
try:
    if not ensure_model_file():
        raise RuntimeError("Model not available")

    model = keras.models.load_model(MODEL_PATH, compile=False)
    print(f"[OK] Loaded model: {MODEL_PATH}")
    print(f"[OK] Num classes: {len(CLASS_LABELS)}")
except Exception as e:
    print(f"[ERR] Cannot load model: {e}")
    model = None


# ===================== UTILS =====================
def preprocess_image(img: Image.Image, target_size=(224, 224), apply_enhancements=False):
    img = img.resize(target_size)

    if apply_enhancements:
        img = ImageEnhance.Contrast(img).enhance(1.5)
        img = ImageEnhance.Brightness(img).enhance(1.2)
        img = img.filter(ImageFilter.SHARPEN)

    arr = np.array(img, dtype=np.float32)

    # Nếu lỡ có alpha channel
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[:, :, :3]

    arr = np.expand_dims(arr, axis=0) / 255.0
    return arr


def get_prediction(img_array, confidence_threshold=98):
    if model is None:
        return "Model not loaded", 0.0

    preds = model.predict(img_array, verbose=0)
    idx = int(np.argmax(preds, axis=1)[0])

    if idx < 0 or idx >= len(CLASS_LABELS):
        return "Unknown class index", 0.0

    label = CLASS_LABELS[idx]
    confidence = float(preds[0][idx]) * 100.0

    if confidence < confidence_threshold:
        return "not found", confidence

    return label, confidence


# ===================== ROUTES =====================
@app.route("/", methods=["GET"])
def health():
    return "OK", 200


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model không khả dụng trên server"}), 503

    data = request.get_json(silent=True) or {}
    if "image" not in data:
        return jsonify({"error": 'Thiếu trường "image" trong request'}), 400

    try:
        img_bytes = base64.b64decode(data["image"])
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        img_type = data.get("image_type", "file")
        use_enhancements = (img_type == "screenshot")

        img_array = preprocess_image(img, apply_enhancements=use_enhancements)
        label, confidence = get_prediction(img_array, confidence_threshold=CONFIDENCE_THRESHOLD)

        return jsonify({
            "label": str(label),
            "confidence": float(confidence)
        }), 200

    except (base64.binascii.Error, IOError):
        return jsonify({"error": "Dữ liệu ảnh không hợp lệ hoặc bị hỏng"}), 400
    except Exception as e:
        print(f"[ERR] /predict error: {e}")
        return jsonify({"error": "Đã xảy ra lỗi không xác định trên server"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
