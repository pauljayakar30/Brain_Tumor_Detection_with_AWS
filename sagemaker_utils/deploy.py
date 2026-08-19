import os
import sagemaker
from sagemaker.tensorflow.model import TensorFlowModel

sess = sagemaker.Session()
role = f"arn:aws:iam::{os.environ['AWS_ACCOUNT_ID']}:role/SageMakerExecutor"

model = TensorFlowModel(
    model_data="s3://project-artifacts-0401/model.tar.gz",
    role=role,
    entry_point="inference.py",
    source_dir="sagemaker_utils",
    framework_version="2.10",
)

predictor = model.deploy(
    instance_type="ml.m5.large",
    initial_instance_count=1,
    endpoint_name="brain-tumor-tf-endpoint"
)