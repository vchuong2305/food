import os
import io
import base64
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

from flask import Flask, request, jsonify
from flask_cors import CORS

from tensorflow import keras
import gdown



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cnn50_100_16_4.h5")
# set trên Render env, ví dụ: MODEL_FILE_ID=1AbC...
MODEL_FILE_ID = os.environ.get("MODEL_FILE_ID", "").strip()

def ensure_model():
    if os.path.exists(MODEL_PATH):
        print("[OK] Model already exists locally.")
        return True

    if not MODEL_FILE_ID:
        print("[ERR] Missing MODEL_FILE_ID env.")
        return False

    url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
    print("[INFO] Downloading model from Google Drive...")
    gdown.download(url, MODEL_PATH, quiet=False)
    ok = os.path.exists(MODEL_PATH)
    print("[OK] Download done." if ok else "[ERR] Download failed.")
    return ok

CLASS_LABELS = [
    "Banh beo", "Banh chung", "Banh cuon", "Banh mi",
    "Banh trang nuong", "Banh xeo", "Bun dau mam tom",
    "Ca kho to", "Pho", "Xoi xeo"
]

CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", 98))


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


model = None
try:
    if not ensure_model():
        raise RuntimeError("Model not available")
    
    model = keras.models.load_model(MODEL_PATH, compile=False)
    print(f"[OK] Loaded model: {MODEL_PATH}")
    print(f"[OK] Num classes: {len(CLASS_LABELS)}")
except Exception as e:
    print(f"[ERR] Cannot load model: {e}")
    model = None


def preprocess_image(img: Image.Image, target_size=(224, 224), apply_enhancements=False):
    img = img.resize(target_size)

    if apply_enhancements:
        img = ImageEnhance.Contrast(img).enhance(1.5)
        img = ImageEnhance.Brightness(img).enhance(1.2)
        img = img.filter(ImageFilter.SHARPEN)

    img_array = np.array(img, dtype=np.float32)

    if img_array.ndim == 3 and img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array


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


# ====== ROUTES ======
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
        img_data = base64.b64decode(data["image"])
        img = Image.open(io.BytesIO(img_data)).convert("RGB")

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
    # ✅ Render/Cloud dùng PORT env
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
