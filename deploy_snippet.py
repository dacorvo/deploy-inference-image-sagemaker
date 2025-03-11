import boto3
import os
from sagemaker.huggingface import HuggingFaceModel

#role = get_execution_role()
iam = boto3.client("iam")
role = iam.get_role(RoleName="sagemaker_execution_role")["Role"]["Arn"]

# Hub Model configuration. https://huggingface.co/models
hub = {
    "HF_MODEL_ID": "meta-llama/Llama-3.2-1B-Instruct",
    "HF_NUM_CORES": "2",
    "HF_AUTO_CAST_TYPE": "bf16",
    "MAX_BATCH_SIZE": "1",
    "MAX_INPUT_TOKENS": "3686",
    "MAX_TOTAL_TOKENS": "4096",
    "HF_TOKEN": os.environ["HF_TOKEN"],
}

region = boto3.Session().region_name
image_uri = f"763104351884.dkr.ecr.{region}.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.2-optimum0.0.27-neuronx-py310-ubuntu22.04"

# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
    image_uri=image_uri,
    env=hub,
    role=role,
)

# deploy model to SageMaker Inference
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.inf2.xlarge",
    container_startup_health_check_timeout=1800,
    volume_size=512,
)

# send request
output = predictor.predict(
    {
        "inputs": "Write a function to generate random numbers in python",
        "parameters": {
            "do_sample": True,
            "max_new_tokens": 256,
            "temperature": 0.1,
            "top_k": 10,
        }
    }
)
print(output)
