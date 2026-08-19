import os
import tarfile

import boto3
import sagemaker
from sagemaker.tensorflow import TensorFlow

# ── Config ────────────────────────────────────────────────────────────────────

ROLE        = "arn:aws:iam::494825111473:role/SageMakerExecutor"
S3_BUCKET   = "project-artifacts-2204"
REGION      = os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")
DATASET_DIR = "dataset"
DATASET_ARCHIVE = "dataset.tar.gz"
S3_DATASET_KEY  = "dataset/dataset.tar.gz"


EPOCHS     = 100
BATCH_SIZE = 42
INSTANCE   = "ml.m5.xlarge"

def upload_dataset():
    print(f"Packaging '{DATASET_DIR}/' → '{DATASET_ARCHIVE}'...")
    with tarfile.open(DATASET_ARCHIVE, "w:gz") as tar:
        for split in ("yes", "no"):
            split_path = os.path.join(DATASET_DIR, split)
            if os.path.isdir(split_path):
                tar.add(split_path, arcname=split)
            else:
                print(f"  WARNING: '{split_path}' not found, skipping.")

    s3 = boto3.client("s3", region_name=REGION)
    size_mb = os.path.getsize(DATASET_ARCHIVE) / 1024 / 1024
    print(f"Uploading ({size_mb:.2f} MB) → s3://{S3_BUCKET}/{S3_DATASET_KEY}")
    s3.upload_file(DATASET_ARCHIVE, S3_BUCKET, S3_DATASET_KEY)
    s3_uri = f"s3://{S3_BUCKET}/{S3_DATASET_KEY}"
    print(f"Dataset uploaded: {s3_uri}")
    return s3_uri


def launch_training(dataset_s3_uri):
    sess = sagemaker.Session(boto_session=boto3.Session(region_name=REGION))

    estimator = TensorFlow(
        entry_point="sm_train_entry.py",
        source_dir="utilities",
        role=ROLE,
        instance_type=INSTANCE,
        instance_count=1,
        framework_version="2.10",
        py_version="py39",
        hyperparameters={
            "epochs":     EPOCHS,
            "batch-size": BATCH_SIZE,
        },
        output_path=f"s3://{S3_BUCKET}/training-jobs/",
        sagemaker_session=sess,
    )

    print("Submitting training job...")
    estimator.fit({"training": dataset_s3_uri}, wait=True, logs="All")

    model_artifact = estimator.model_data
    print(f"\nTraining complete.")
    print(f"Model artifact: {model_artifact}")
    return model_artifact


if __name__ == "__main__":
    dataset_uri = upload_dataset()
    launch_training(dataset_uri)
