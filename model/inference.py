import base64
import io
import json
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

CLASS_NAMES = ["No", "Yes"]

def model_fn(model_dir):
    model = load_model(model_dir)
    return model

def input_fn(request_body, content_type):
    if content_type == "application/json":
        body = json.loads(request_body)
        image_bytes = base64.b64decode(body["inputs"])
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    elif content_type in ("image/jpeg", "image/png"):
        image = Image.open(io.BytesIO(request_body)).convert('RGB')
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

    image = image.resize((128, 128))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    return image


def predict_fn(input_data, model):
    predictions = model.predict(input_data)[0]
    class_id = int(np.argmax(predictions))
    return {
        "class": CLASS_NAMES[class_id],
        "confidence": float(predictions[class_id])
    }

def output_fn(prediction, accept):
    if accept in ("application/json", "*/*"):
        return json.dumps(prediction), "application/json"
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
