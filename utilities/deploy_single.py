import os

from sagemaker.huggingface import HuggingFaceModel

role = f"arn:aws:iam::{os.environ['AWS_ACCOUNT_ID']}:role/SageMakerExecutor"

model = HuggingFaceModel(
    transformers_version="4.37",
    pytorch_version="2.1",
    py_version="py310",
    env = {
        "HF_MODEL_ID": "Devarshi/Brain_Tumor_Classification",
        "HF_TASK": "image-classification"
    },
    role=role
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name="brain-tumor-endpoint"
)

print(predictor)