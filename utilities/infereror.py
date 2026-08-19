import boto3
import json
import numpy as np
from PIL import Image

runtime = boto3.client("sagemaker-runtime", region_name="ap-south-1")

CLASS_NAMES = ["Tumor Detected", "No Tumor"]

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB").resize((128, 128))
    arr = np.array(img, dtype=np.float32)
    payload = json.dumps({"instances": [arr.tolist()]})

    response = runtime.invoke_endpoint(
        EndpointName="2304-tumor-endpoint",
        ContentType="application/json",
        Accept="application/json",
        Body=payload
    )

    body = json.loads(response['Body'].read())
    probs = body["predictions"][0]
    class_id = int(np.argmax(probs))
    return {
        "result": CLASS_NAMES[class_id],
        "confidence": f"{probs[class_id] * 100:.1f}%"
    }

if __name__ == "__main__":
    print(predict_image("./dataset/yes/Y1.jpg"))