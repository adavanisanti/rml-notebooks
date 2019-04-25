from rocketml.dbutils import mount_s3_bucket,unmount_s3_bucket
import os
access_key = os.environ.get("AWS_ACCESS_KEY","")
secret_key = os.environ.get("AWS_SECRET_KEY","")
bucket_name = "rocketml-dev"
mount_dir = "/home/ubuntu/rocketml-dev"
region="us-west-2"
mount_s3_bucket(access_key=access_key,secret_key=secret_key,bucket_name=bucket_name,mount_dir=mount_dir,region=region)
#unmount_s3_bucket(mount_dir=mount_dir)
