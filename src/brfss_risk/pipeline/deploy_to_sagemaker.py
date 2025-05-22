from sagemaker import Model, Session
from sagemaker.image_uris import retrieve
import boto3, datetime as dt

region   = "us-east-1"
bucket   = ""
s3_key   = ""
role     = ""
endpoint = f"xgb-native-{dt.datetime.now():%Y%m%d-%H%M}"
image_uri = retrieve("xgboost", region=region, version="1.5-1")

print(f"Starting deployment to endpoint: {endpoint}")
print(f"Model S3 path: s3://{bucket}/{s3_key}")
print(f"Using container image URI: {image_uri}")

# Set up SageMaker session
sm_sess = Session(boto_session=boto3.Session(region_name=region))

# Define model
model = Model(
    image_uri=image_uri,
    model_data=f"s3://{bucket}/{s3_key}",
    role=role,
    sagemaker_session=sm_sess
)
try:
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.large",
        endpoint_name=endpoint,
        wait=True,
        logs=True
    )
    print("Deployment complete")
    print("Deployed endpoint →", endpoint)
except Exception as e:
    print("Deployment failed")
    print(e)

