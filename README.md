# 🧠 Brain Tumor Detection with AWS SageMaker

An end-to-end deep learning pipeline that classifies brain MRI scans as tumor / no-tumor — covering local model training, cloud deployment on AWS SageMaker, and a Streamlit web app for real-time inference.

## Overview

This project takes an MRI image, runs it through a custom CNN, and returns a prediction with a confidence score. The codebase includes two training paths — a self-contained local script and a SageMaker training-container script — but **inference is AWS-only**: predictions require a live SageMaker endpoint, since the inference logic is written as SageMaker handler functions rather than a standalone local predictor.

## Architecture

**Training**: `utilities/build_model.py` defines the CNN architecture. Two scripts can drive training with it: `utilities/trainer.py` (a self-contained local script — no AWS involved) and `utilities/sm_train_entry.py` (the equivalent entrypoint that runs inside a SageMaker training container, launched via `utilities/launch_training.py`). Both build the same architecture —
- Two convolutional blocks (32 and 64 filters, with BatchNorm, MaxPooling, and Dropout for regularization)
- A dense classification head (512 units → softmax over 2 classes)
- Trained with the Adamax optimizer and categorical cross-entropy loss
- Input images resized to 128×128×3

**Inference — AWS-only**: Unlike training, there's no standalone local inference path. `model/inference.py` and `sagemaker_utils/inference.py` define SageMaker handler functions (`model_fn`, `input_fn`, `predict_fn`, `output_fn`) that run *inside* a live SageMaker endpoint — they aren't meant to be called directly outside that environment.

**Deployment**: The model is deployed to a SageMaker endpoint via `sagemaker_utils/deploy.py` or `utilities/deploy_own_model.py`. The Streamlit app queries this live endpoint via `boto3` for predictions.

**Web app**: `webapp/app.py` — built with Streamlit. Upload an MRI image, and it's sent to the SageMaker endpoint, which returns a classification (Tumor Detected / No Tumor) with a confidence percentage.

## Project Structure

```
├── model/
│   └── inference.py          # SageMaker inference handlers (runs inside the endpoint)
├── sagemaker_utils/
│   ├── deploy.py               # Deploys the custom CNN (from S3) to a SageMaker endpoint
│   ├── inference.py            # SageMaker inference handlers used by deploy.py's endpoint
│   └── requirements.txt        # Minimal deps (Pillow, numpy) bundled with the deployed endpoint
├── utilities/
│   ├── build_model.py          # CNN architecture definition
│   ├── data_preparation.py     # Loads and preprocesses MRI images from disk
│   ├── trainer.py               # Local training driver (build + fit + save)
│   ├── sm_train_entry.py        # Training entrypoint that runs inside the SageMaker training container
│   ├── launch_training.py       # Packages the dataset and kicks off a SageMaker training job
│   ├── deploy_own_model.py      # Deploys the custom-trained CNN to a SageMaker endpoint
│   ├── deploy_single.py         # Deploys a pretrained third-party model from Hugging Face (Devarshi/Brain_Tumor_Classification) — a separate experiment, not the custom CNN
│   ├── infereror.py             # CLI script to test predictions against a live endpoint
│   ├── upload_model_s3.py       # Packages the locally trained model and uploads it to S3
│   └── cleanup.py               # Tears down the SageMaker endpoint, config, models, and S3 artifacts
├── webapp/
│   └── app.py                   # Streamlit frontend for live predictions
└── requirements.txt
```

## Setup

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set required environment variables (see `.gitignore` — a `set_environment.sh` script is expected but not tracked, since it holds AWS credentials/account details):
   ```bash
   export AWS_ACCOUNT_ID="your-account-id"
   export AWS_DEFAULT_REGION="your-region"
   ```

3. **Dataset**: place MRI images under `dataset/yes/` (tumor present) and `dataset/no/` (no tumor). Images are auto-resized to 128×128 during preprocessing.

## Training & Deployment

- **Local training** (no AWS required): `utilities/data_preparation.py` prepares the dataset, `utilities/trainer.py` builds and trains the model, saving it to `trained_model/`.
- **SageMaker training**: `utilities/launch_training.py` packages the dataset, uploads it to S3, and launches a training job running `utilities/sm_train_entry.py` inside a SageMaker container.
- **Deploy to an endpoint**: `sagemaker_utils/deploy.py` or `utilities/deploy_own_model.py` — both deploy the custom-trained CNN. (`utilities/deploy_single.py` is a separate path that deploys a pretrained Hugging Face model instead, not the custom CNN.)
- **Test the endpoint from the command line**: `utilities/infereror.py` sends a sample image to a live endpoint and prints the prediction.
- **Tear down** when done (to avoid ongoing AWS costs): `utilities/cleanup.py` removes the endpoint, its config, associated models, and S3 artifacts.

## Running the Web App

Once a SageMaker endpoint is live:
```bash
streamlit run webapp/app.py
```
Upload an MRI scan and the app returns a real-time classification with a confidence score.

## Notes

- AWS credentials and account-specific values are kept out of version control via `.gitignore` and environment variables — see `set_environment.sh` (not tracked) for local setup.
- This is a learning/portfolio project; SageMaker endpoints incur ongoing AWS costs while active, so remember to run `cleanup.py` after testing.

## Quick Reference — All Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables (put these in set_environment.sh, then `source` it)
export AWS_ACCOUNT_ID="your-account-id"
export AWS_DEFAULT_REGION="your-region"

# 3a. Train locally (no AWS)
python -m utilities.trainer

# 3b. OR train on SageMaker
python -m utilities.launch_training

# 4. Package + upload a locally trained model to S3 (only needed if you trained locally)
python -m utilities.upload_model_s3

# 5. Deploy the custom CNN to a SageMaker endpoint (pick one)
python -m utilities.deploy_own_model      # endpoint: 2304-tumor-endpoint (used by the webapp)
python -m sagemaker_utils.deploy          # endpoint: brain-tumor-tf-endpoint (alt/unused by webapp)

# 5b. OR deploy the separate Hugging Face model experiment (not the custom CNN)
python -m utilities.deploy_single         # endpoint: brain-tumor-endpoint

# 6. Test the live endpoint from the command line
python -m utilities.infereror

# 7. Run the web app (needs the 2304-tumor-endpoint live)
streamlit run webapp/app.py

# 8. Tear down AWS resources when done, to stop incurring costs
python -m utilities.cleanup
```

## License

MIT