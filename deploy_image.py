import argparse
import os
import time
import boto3
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.compute_resource_requirements.resource_requirements import ResourceRequirements
from typing import Dict


def deploy_image(image: str,
                 config: Dict[str, str],
                 instance_type: str,
                 iam_role: str):
    start = time.time()
    iam = boto3.client("iam")
    role = iam.get_role(RoleName=iam_role)["Role"]["Arn"]

    print(f"sagemaker role arn: {role}")
    print(f"instance type: {instance_type}")
    print(f"config: {config}")

    # create HuggingFaceModel
    llm_model = HuggingFaceModel(role=role, image_uri=image, env=config)

    resources_config = ResourceRequirements(
        # We dedicate 12 vCPU and 18 GB memory for the sagemaker wrapper
        requests = {
            "copies": 3, # Number of replicas
            "num_accelerators": 1, # Number of accelerators (devices) per replica
            "num_cpus": 60,  # Number of CPU cores per replica (192 - 12) // 3 = 60
            "memory": 250 * 1024,  # Minimum memory in MB per replica (768 - 18) // 3 = 250
        },
    )

    # deploy model to endpoint
    try:
        llm = llm_model.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            container_startup_health_check_timeout=1800, # Neuron models take a long time to load + warmup
            volume_size=256,
            inference_ami_version = "al2-ami-sagemaker-inference-neuron-2",
            resources=resources_config,
        )
        print(f"Successfully deployed {llm_model.name} as endpoint {llm_model.endpoint_name}")
    except Exception as e:
        print(e)
        print(f"Failed to deploy model with config {config} on {instance_type}")
    finally:
        print(f"Total time: {round(time.time() - start)}s")


# TGI deployment config
def get_neuronx_tgi_config(model_id, batch_size, sequence_length, auto_cast_type, num_cores, token):

    max_input_length = sequence_length // 2
    max_total_tokens = sequence_length
    max_batch_prefill_tokens = batch_size * max_input_length
    max_batch_total_tokens = batch_size * sequence_length

    tgi_config = {
        "HF_MODEL_ID": model_id,
        "HF_NUM_CORES": f"{num_cores}",
        "HF_BATCH_SIZE": f"{batch_size}",
        "HF_SEQUENCE_LENGTH": f"{sequence_length}",
        "HF_AUTO_CAST_TYPE": auto_cast_type,
        "MAX_BATCH_SIZE": f"{batch_size}",
        "MAX_CONCURRENT_REQUESTS": "128",
        "MAX_INPUT_LENGTH": f"{max_input_length}",
        "MAX_TOTAL_TOKENS": f"{max_total_tokens}",
        "MAX_BATCH_PREFILL_TOKENS": f"{max_batch_prefill_tokens}",
        "MAX_BATCH_TOTAL_TOKENS": f"{max_batch_total_tokens}",
    }

    if token:
        tgi_config["HUGGING_FACE_HUB_TOKEN"] = token

    return tgi_config


# vLLM deployment config
def get_neuronx_vllm_config(model_id, batch_size, sequence_length, auto_cast_type, num_cores, token):

    vllm_config = {
        "SM_VLLM_MODEL": model_id,
        "SM_VLLM_MAX_NUM_SEQS": f"{batch_size}",
        "SM_VLLM_TENSOR_PARALLEL_SIZE": f"{num_cores}",
        "SM_VLLM_MAX_MODEL_LEN": f"{sequence_length}",
    }

    if token:
        vllm_config["HF_TOKEN"] = token

    return vllm_config


if __name__ == "__main__":
    # Query the current region
    session = boto3.session.Session()
    current_region = session.region_name

    parser = argparse.ArgumentParser(description="Deploy a service based on an HuggingFace image on sagemaker")
    parser.add_argument("--image", type=str, required=True, help="The full Sagemaker image URI")
    parser.add_argument("--model_id", type=str, required=True, help="The HuggingFace model id")
    parser.add_argument("--instance_type", type=str, required=True, help="The Sagemaker trainium/inferentia instance type")
    parser.add_argument("--iam_role", default="sagemaker_execution_role", type=str)
    parser.add_argument("--region",
                        type=str,
                        default="us-east-1" if current_region is None else current_region)
    parser.add_argument("--token",
                        type=str,
                        help="The HuggingFace token to use to fetch the model if gated or private.",
                        default=os.environ.get("HF_TOKEN", None))
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size.",
    )
    parser.add_argument("--sequence_length", type=int, help="The maximum sequence length.")
    parser.add_argument(
        "--num_cores", type=int, default=2, help="The number of cores on which the model should be split."
    )
    parser.add_argument(
        "--auto_cast_type", type=str, default="bf16", choices=["fp32", "fp16", "bf16"], help="One of fp32, fp16, bf16."
    )
    args = parser.parse_args()

    # Set region
    boto3.setup_default_session(region_name=args.region)

    image = args.image
    if not "amazonaws.com" in image:
        raise ValueError("You need to pass a full Sagemaker image URI")

    if "vllm" in image:
        config = get_neuronx_vllm_config(args.model_id,
                                        args.batch_size,
                                        args.sequence_length,
                                        args.auto_cast_type,
                                        args.num_cores,
                                        args.token)
    elif "tgi" in image:
        config = get_neuronx_tgi_config(args.model_id,
                                        args.batch_size,
                                        args.sequence_length,
                                        args.auto_cast_type,
                                        args.num_cores,
                                        args.token)
    else:
        raise ValueError("You must pass a TGI or vLLM image")

    deploy_image(image,
                 config,
                 instance_type=args.instance_type,
                 iam_role=args.iam_role)
