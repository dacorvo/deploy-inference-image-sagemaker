import boto3
import os
import uuid
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri
from sagemaker.compute_resource_requirements.resource_requirements import ResourceRequirements
from sagemaker.enums import EndpointType

iam = boto3.client("iam")
role = iam.get_role(RoleName="sagemaker_execution_role")["Role"]["Arn"]

# Hub Model configuration. https://huggingface.co/models
hub = {
    "HF_MODEL_ID": "meta-llama/Llama-2-7b-chat-hf",
    "HF_NUM_CORES": "8",
    "HF_AUTO_CAST_TYPE": "fp16",
    "MAX_BATCH_SIZE": "32",
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

# Set resources

resources_config = ResourceRequirements(
    # We dedicate 12 vCPU and 18 GB memory for the sagemaker wrapper
    requests = {
        "copies": 3, # Number of replicas
        "num_accelerators": 4, # Number of accelerators (devices) per replica
        "num_cpus": 60,  # Number of CPU cores per replica (192 - 12) // 3 = 60
        "memory": 250 * 1024,  # Minimum memory in MB per replica (768 - 18) // 3 = 250
    },
)

# deploy model to SageMaker Inference
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.inf2.48xlarge",
    container_startup_health_check_timeout=3600,
    volume_size=512,
    resources=resources_config,
    endpoint_name=f"llama-3-8B-DP3TP8-{str(uuid.uuid4())}",
    endpoint_type=EndpointType.INFERENCE_COMPONENT_BASED
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
