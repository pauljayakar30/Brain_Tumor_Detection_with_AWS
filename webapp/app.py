import streamlit as st
import boto3
import os
import json
import numpy as np
from PIL import Image
import io

ENDPOINT_NAME = "2304-tumor-endpoint"
CLASS_NAMES   = ["Tumor Detected", "No Tumor"]
REGION        = os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")


st.set_page_config(
    page_title="Brain Tumor Detector",
    page_icon="🧠",
    layout="centered",
)


st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .block-container { padding-top: 2rem; }
    .title  { text-align: center; font-size: 2.4rem; font-weight: 700;
              color: #ffffff; margin-bottom: 0.2rem; }
    .sub    { text-align: center; color: #9ca3af; font-size: 1rem;
              margin-bottom: 2rem; }
    .result-card {
        border-radius: 12px; padding: 1.5rem 2rem;
        text-align: center; margin-top: 1.5rem;
    }
    .result-positive { background: linear-gradient(135deg,#ff4b4b22,#ff4b4b44);
                       border: 1px solid #ff4b4b88; }
    .result-negative { background: linear-gradient(135deg,#21c45d22,#21c45d44);
                       border: 1px solid #21c45d88; }
    .result-label  { font-size: 1.8rem; font-weight: 700; margin-bottom: 0.3rem; }
    .result-conf   { font-size: 1rem; color: #d1d5db; }
    .positive-text { color: #ff4b4b; }
    .negative-text { color: #21c45d; }
    div[data-testid="stImage"] img { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


st.markdown('<p class="title">🧠 Brain Tumor Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="sub">Upload an MRI scan and the model will classify it instantly using AWS SageMaker.</p>', unsafe_allow_html=True)


def predict_image(file_bytes: bytes):
    runtime = boto3.client("sagemaker-runtime", region_name=REGION)
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB").resize((128, 128))
    arr = np.array(img, dtype=np.float32)
    payload = json.dumps({"instances": [arr.tolist()]})

    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=payload,
    )
    body  = json.loads(response["Body"].read().decode("utf-8"))
    probs = body["predictions"][0]
    class_id = int(np.argmax(probs))
    return CLASS_NAMES[class_id], float(probs[class_id]) * 100


col_upload, col_preview = st.columns([1.2, 1], gap="large")

with col_upload:
    st.markdown("#### Upload MRI Image")
    uploaded_file = st.file_uploader(
        "Supported formats: JPG, JPEG, PNG",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    submit = st.button("Analyze", use_container_width=True, type="primary")

with col_preview:
    if uploaded_file:
        st.markdown("#### Preview")
        st.image(uploaded_file, use_container_width=True)


if submit:
    if uploaded_file is None:
        st.warning("Please upload an MRI image before clicking Analyze.")
    else:
        with st.spinner("Running inference on SageMaker..."):
            try:
                file_bytes = uploaded_file.getvalue()
                label, confidence = predict_image(file_bytes)
                is_positive = label == "Tumor Detected"
                card_class  = "result-positive" if is_positive else "result-negative"
                text_class  = "positive-text"   if is_positive else "negative-text"
                icon        = "🔴" if is_positive else "🟢"

                st.markdown(f"""
                <div class="result-card {card_class}">
                    <div class="result-label {text_class}">{icon} {label}</div>
                    <div class="result-conf">Confidence: <strong>{confidence:.1f}%</strong></div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("##### Probability Breakdown")
                st.progress(confidence / 100,
                            text=f"{label} — {confidence:.1f}%")
            except Exception as e:
                st.error(f"Inference failed: {e}")