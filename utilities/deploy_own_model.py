import boto3
import sagemaker
from sagemaker.tensorflow.model import TensorFlowModel

ENDPOINT_NAME = "2304-tumor-endpoint"
REGION = "ap-south-1"

sess = sagemaker.Session()
role = "arn:aws:iam::494825111473:role/SageMakerExecutor"
sm = boto3.client("sagemaker", region_name=REGION)


def cleanup():
    for delete_fn, resource, name_key in [
        (sm.delete_endpoint,        "endpoint",        "EndpointName"),
        (sm.delete_endpoint_config, "endpoint config", "EndpointConfigName"),
    ]:
        try:
            delete_fn(**{name_key: ENDPOINT_NAME})
            print(f"Deleted existing {resource}: {ENDPOINT_NAME}")
        except sm.exceptions.from_code("ValidationException"):
            pass
        except Exception:
            pass


cleanup()

model = TensorFlowModel(
    model_data="s3://project-artifacts-2304/model.tar.gz",
    role=role,
    framework_version="2.10",
    sagemaker_session=sess,
)

predictor = model.deploy(
    instance_type="ml.m5.large",
    initial_instance_count=1,
    endpoint_name=ENDPOINT_NAME,
)

print(predictor)
