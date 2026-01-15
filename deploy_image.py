import argparse
import os
import time
import boto3
import warnings
from sagemaker.core.resources import Endpoint, Model, ContainerDefinition, EndpointConfig, ProductionVariant


def deploy_image(image: str,
                 model_id: str,
                 instance_type: str,
                 batch_size: int | None = None,
                 sequence_length: int | None = None,
                 tensor_parallel_size: int | None = None,
                 token: str | None = None,
                 iam_role: str = "sagemaker_execution_role"):
    start = time.time()
    iam = boto3.client("iam")
    role = iam.get_role(RoleName=iam_role)["Role"]["Arn"]

    print(f"sagemaker role arn: {role}")
    print(f"instance type: {instance_type}")

    # vLLM deployment parameters are specified in the environment variables
    environment = {
        "SM_ON_MODEL": model_id,
    }
    if batch_size is not None:
        environment["SM_ON_MAX_NUM_SEQS"] = str(batch_size)
    if sequence_length is not None:
        environment["SM_ON_MAX_MODEL_LEN"] = str(sequence_length)
    if tensor_parallel_size is not None:
        environment["SM_ON_TENSOR_PARALLEL_SIZE"] = str(tensor_parallel_size)
    if token is not None:
        environment["HF_TOKEN"] = token

    print(f"deployment parameters: {environment}")

    container = ContainerDefinition(image=image, environment=environment)

    model_name = model_id.split("/")[-1].replace(".", "-").lower()

    endpoint_name = model_name + "-vllm-neuron-" + str(int(time.time()))

    model = Model.create(
        model_name=endpoint_name,
        primary_container=container,
        execution_role_arn=role,
    )
    assert model is not None

    volume_size = None
    if not "trn1" in instance_type:
        # With most instance types a separate volume is mounted dynamically under /tmp.
        # This volume has by default only 50B of disk space, so it needs to be increased
        # to support large models.
        # Trainium 1 endpoints do not have this limitation because each Trainium instance
        # comes with 4 disk drives of fixed size. As a consequence, the volume_size parameter
        # is not supported.
        volume_size = 256
    endpoint_config = EndpointConfig.create(
        endpoint_config_name=endpoint_name,
        production_variants=[
            ProductionVariant(
                variant_name="AllTraffic",
                model_name=model.model_name,
                initial_instance_count=1,
                instance_type=instance_type,
                container_startup_health_check_timeout_in_seconds=1800, # Neuron models take a long time to load + warmup
                volume_size_in_gb=volume_size,
                inference_ami_version = "al2-ami-sagemaker-inference-neuron-2"
            )
        ],
    )
    assert endpoint_config is not None

    endpoint = Endpoint.create(
        endpoint_name=endpoint_name,
        endpoint_config_name=endpoint_config.endpoint_config_name,
    )
    assert endpoint is not None

    endpoint.wait_for_status(target_status='InService')
    print(f"Successfully deployed {model.model_name} as endpoint {endpoint.endpoint_name}")
    print(f"Total time: {round(time.time() - start)}s")


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
    parser.add_argument("--batch_size", type=int, help="The batch size.")
    parser.add_argument("--sequence_length", type=int, help="The maximum sequence length.")
    parser.add_argument("--tensor_parallel_size", type=int, help="The number of cores on which the model should be split.")
    args = parser.parse_args()

    # Set region
    boto3.setup_default_session(region_name=args.region)

    image = args.image
    if not "amazonaws.com" in image:
        raise ValueError("You need to pass a full Sagemaker image URI")

    if args.token is None:
        warnings.warn("You did not pass a HuggingFace token. Make sure the model is public."
                      "Please note also that your endpoint will be rate limited when fetching"
                      "from the Hugging Face hub and may not be able to start.")

    deploy_image(image,
                 model_id=args.model_id,
                 batch_size=args.batch_size,
                 sequence_length=args.sequence_length,
                 tensor_parallel_size=args.tensor_parallel_size,
                 token=args.token,
                 instance_type=args.instance_type,
                 iam_role=args.iam_role)
