import os
import boto3

ENDPOINT_NAME = "2304-tumor-endpoint"
S3_BUCKET     = "project-artifacts-2304"
S3_KEY        = "model.tar.gz"
REGION        = os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")

sm = boto3.client("sagemaker", region_name=REGION)
s3 = boto3.client("s3",        region_name=REGION)


def _try(label, fn, **kwargs):
    try:
        fn(**kwargs)
        print(f"  [OK]      Deleted {label}")
    except sm.exceptions.from_code("ResourceNotFound"):
        print(f"  [SKIP]    {label} not found")
    except sm.exceptions.from_code("ValidationException"):
        print(f"  [SKIP]    {label} not found")
    except Exception as e:
        print(f"  [ERROR]   {label}: {e}")


def delete_endpoint():
    print("Deleting endpoint...")
    _try(f"endpoint '{ENDPOINT_NAME}'",
         sm.delete_endpoint, EndpointName=ENDPOINT_NAME)


def delete_endpoint_config():
    print("Deleting endpoint config...")
    _try(f"endpoint config '{ENDPOINT_NAME}'",
         sm.delete_endpoint_config, EndpointConfigName=ENDPOINT_NAME)


def delete_models():
    print("Deleting SageMaker model(s)...")
    paginator = sm.get_paginator("list_models")
    deleted = 0
    for page in paginator.paginate(NameContains=ENDPOINT_NAME):
        for m in page["Models"]:
            name = m["ModelName"]
            _try(f"model '{name}'", sm.delete_model, ModelName=name)
            deleted += 1
    if deleted == 0:
        print(f"  [SKIP]    no models found matching '{ENDPOINT_NAME}'")


def delete_s3_artifact():
    print(f"Deleting s3://{S3_BUCKET}/{S3_KEY}...")
    try:
        s3.delete_object(Bucket=S3_BUCKET, Key=S3_KEY)
        print(f"  [OK]      Deleted s3://{S3_BUCKET}/{S3_KEY}")
    except s3.exceptions.from_code("NoSuchKey"):
        print(f"  [SKIP]    s3://{S3_BUCKET}/{S3_KEY} not found")
    except Exception as e:
        print(f"  [ERROR]   S3 delete: {e}")


def delete_local_archive():
    print("Deleting local model.tar.gz...")
    if os.path.exists("model.tar.gz"):
        os.remove("model.tar.gz")
        print("  [OK]      Deleted local model.tar.gz")
    else:
        print("  [SKIP]    local model.tar.gz not found")


if __name__ == "__main__":
    print("=" * 50)
    print("  Brain Tumor Project — AWS Cleanup")
    print("=" * 50)
    delete_endpoint()
    delete_endpoint_config()
    delete_models()
    delete_s3_artifact()
    delete_local_archive()
    print("=" * 50)
    print("  Cleanup complete.")
    print("=" * 50)
