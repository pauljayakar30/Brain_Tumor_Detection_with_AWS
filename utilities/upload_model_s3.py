import os
import tarfile
import boto3

TRAINED_MODEL_DIR = "trained_model"
ARCHIVE_NAME = "model.tar.gz"
S3_BUCKET = "project-artifacts-2304"
S3_KEY = "model.tar.gz"
REGION = os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")


def package_model():
    print(f"Packaging '{TRAINED_MODEL_DIR}' into '{ARCHIVE_NAME}'...")
    with tarfile.open(ARCHIVE_NAME, "w:gz") as tar:
        for item in os.listdir(TRAINED_MODEL_DIR):
            full_path = os.path.join(TRAINED_MODEL_DIR, item)
            tar.add(full_path, arcname=f"1/{item}")
    print(f"Archive created: {ARCHIVE_NAME}")


def upload_to_s3():
    s3 = boto3.client("s3", region_name=REGION)
    file_size = os.path.getsize(ARCHIVE_NAME)
    print(f"Uploading '{ARCHIVE_NAME}' ({file_size / 1024 / 1024:.2f} MB) to s3://{S3_BUCKET}/{S3_KEY}...")

    s3.upload_file(
        ARCHIVE_NAME,
        S3_BUCKET,
        S3_KEY,
    )
    print(f"Upload complete: s3://{S3_BUCKET}/{S3_KEY}")


if __name__ == "__main__":
    package_model()
    upload_to_s3()
