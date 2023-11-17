import time
import boto3
from sagemaker.huggingface import HuggingFaceModel
from typing import Dict


def deploy_image(llm_image: str,
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
    llm_model = HuggingFaceModel(role=role, image_uri=llm_image, env=config)

    # deploy model to endpoint
    try:
        llm = llm_model.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            container_startup_health_check_timeout=600, # Neuron models take a long time to load + warmup
        )
        print(f"Successfully deployed {llm_model.name} as endpoint {llm_model.endpoint_name}")
    except Exception as e:
        print(e)
        print(f"Failed to deploy model with config {config} on {instance_type}")
    finally:
        print(f"Total time: {round(time.time() - start)}s")
