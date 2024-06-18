import boto3
import os
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri

iam = boto3.client("iam")
role = iam.get_role(RoleName="sagemaker_execution_role")["Role"]["Arn"]

# Hub Model configuration. https://huggingface.co/models
hub = {
    "HF_MODEL_ID": "meta-llama/Meta-Llama-3-8B",
    "HF_NUM_CORES": "2",
    "HF_AUTO_CAST_TYPE": "fp16",
    "MAX_BATCH_SIZE": "8",
    "MAX_INPUT_LENGTH": "3686",
    "MAX_TOTAL_TOKENS": "4096",
    "HF_TOKEN": os.environ["HF_TOKEN"],
}

# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
    image_uri=get_huggingface_llm_image_uri("huggingface-neuronx", version="0.0.23"),
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
predictor.predict(
    {
        "inputs": "What is is the capital of France?",
        "parameters": {
            "do_sample": True,
            "max_new_tokens": 128,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.95,
        }
    }
)
