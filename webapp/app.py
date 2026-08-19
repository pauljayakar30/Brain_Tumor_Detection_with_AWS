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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .block-container {
        padding-top: 3rem;
        max-width: 950px;
    }
    
    /* Header section */
    .title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #f0f6fc;
        margin-bottom: 0.4rem;
        letter-spacing: -0.03em;
        text-align: left;
    }
    
    .sub {
        color: #8b949e;
        font-size: 1.05rem;
        margin-bottom: 2.5rem;
        font-weight: 400;
        text-align: left;
    }
    
    /* Alignment styles for headers */
    h3 {
        font-weight: 600 !important;
        font-size: 1.25rem !important;
        color: #f0f6fc !important;
        margin-top: 0px !important;
        margin-bottom: 1.2rem !important;
        border-bottom: 1px solid #21262d;
        padding-bottom: 0.5rem;
    }
    
    /* Upload area customization */
    div[data-testid="stFileUploader"] {
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 0.8rem;
        background-color: #0d1117;
    }
    
    /* Placeholder Box */
    .placeholder-box {
        border: 2px dashed #21262d;
        border-radius: 8px;
        padding: 3rem 1.5rem;
        text-align: center;
        color: #8b949e;
        font-weight: 400;
        font-size: 0.95rem;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background-color: #0d1117;
        margin-top: 0.5rem;
    }
    
    /* Result card styling */
    .result-card {
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        border: 1px solid #21262d;
        background-color: #0d1117;
        margin-top: 1.2rem;
    }
    
    .result-positive {
        border-left: 4px solid #f25f5c;
    }
    
    .result-negative {
        border-left: 4px solid #2a9d8f;
    }
    
    .result-header {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #8b949e;
        margin-bottom: 0.4rem;
        font-weight: 600;
    }
    
    .result-label {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    
    .positive-text {
        color: #f25f5c;
    }
    
    .negative-text {
        color: #2a9d8f;
    }
    
    .result-conf {
        font-size: 0.9rem;
        color: #8b949e;
    }
    
    .result-conf strong {
        color: #c9d1d9;
    }
    
    /* Customization for uploaded images */
    div[data-testid="stImage"] img {
        border-radius: 8px;
        border: 1px solid #21262d;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">Brain Tumor Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="sub">Upload an MRI scan and the model will classify it instantly.</p>', unsafe_allow_html=True)


# Sidebar settings
st.sidebar.title("Inference Settings")
inference_mode = st.sidebar.selectbox(
    "Select Mode",
    ["Local Model (Default)", "AWS SageMaker Endpoint"]
)

if inference_mode == "AWS SageMaker Endpoint":
    st.sidebar.info(f"Using SageMaker endpoint: **{ENDPOINT_NAME}** in region **{REGION}**")
else:
    st.sidebar.success("Running completely locally using Keras. No AWS account or internet required!")

_local_model = None

def get_local_model():
    global _local_model
    if _local_model is None:
        import keras
        import os
        model_path_keras = "trained_model.keras"
        model_path_legacy = "trained_model"
        if os.path.exists(model_path_keras):
            _local_model = keras.models.load_model(model_path_keras)
        elif os.path.exists(model_path_legacy):
            try:
                _local_model = keras.models.load_model(model_path_legacy)
            except Exception:
                try:
                    # TFSMLayer for Keras 3 compatibility with SavedModel format
                    _local_model = keras.layers.TFSMLayer(model_path_legacy, call_endpoint='serving_default')
                except Exception as e:
                    raise FileNotFoundError(f"Failed to load legacy model as TFSMLayer: {e}")
        else:
            raise FileNotFoundError("Local model file ('trained_model.keras' or 'trained_model' directory) not found. Please train the model locally using 'python -m utilities.trainer' first.")
    return _local_model


def predict_image_local(file_bytes: bytes):
    model = get_local_model()
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB").resize((128, 128))
    arr = np.array(img, dtype=np.float32)
    input_data = np.expand_dims(arr, axis=0) # shape (1, 128, 128, 3)

    import keras
    if isinstance(model, keras.layers.TFSMLayer):
        outputs = model(input_data)
        key = list(outputs.keys())[0]
        probs = outputs[key][0].numpy()
    else:
        probs = model.predict(input_data)[0]

    class_id = int(np.argmax(probs))
    return CLASS_NAMES[class_id], float(probs[class_id]) * 100


def predict_image_sagemaker(file_bytes: bytes):
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


col_upload, col_preview = st.columns([1.1, 1], gap="large")

with col_upload:
    st.markdown("### Upload MRI Scan")
    uploaded_file = st.file_uploader(
        "Supported formats: JPG, JPEG, PNG",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    submit = st.button("Analyze Scan", width="stretch", type="primary")
    if submit and uploaded_file is None:
        st.warning("Please upload an MRI image first.")

with col_preview:
    st.markdown("### Visualization & Results")
    if uploaded_file is None:
        st.markdown("""
        <div class="placeholder-box">
            <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#30363d" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom: 1rem;">
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                <circle cx="8.5" cy="8.5" r="1.5"></circle>
                <polyline points="21 15 16 10 5 21"></polyline>
            </svg>
            <div>Awaiting MRI scan upload</div>
            <div style="font-size:0.8rem; color:#484f58; margin-top:0.4rem;">Select an image on the left to begin analysis</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.image(uploaded_file, width="stretch")
        
        if submit:
            with st.spinner(f"Running inference using {inference_mode}..."):
                try:
                    file_bytes = uploaded_file.getvalue()
                    if inference_mode == "Local Model (Default)":
                        label, confidence = predict_image_local(file_bytes)
                    else:
                        label, confidence = predict_image_sagemaker(file_bytes)
                    confidence_str = f"{confidence:.1f}"
                    is_positive = label == "Tumor Detected"
                    card_class  = "result-positive" if is_positive else "result-negative"
                    text_class  = "positive-text"   if is_positive else "negative-text"
                    icon        = "🔴" if is_positive else "🟢"

                    st.markdown(f"""
                    <div class="result-card {card_class}">
                        <div class="result-header">Classification Result</div>
                        <div class="result-label {text_class}">{icon} {label}</div>
                        <div class="result-conf">Confidence: <strong>{confidence_str}%</strong></div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
                    st.progress(confidence / 100, text=f"{label} — {confidence_str}%")
                except Exception as e:
                    st.error(f"Inference failed: {e}")