import argparse
import boto3
from sagemaker.core import image_uris


def retrieve_image_uri(region: str,
                       framework: str | None = "huggingface-vllm-neuronx",
                       version: str | None = None) -> str:
    """
    Retrieve the image URI for a given framework.

    Args:
        framework (str): The framework name (e.g., 'huggingface-vllm-neuronx').
        region (str): The AWS region.
        version (str | None): The version of the framework.

    Returns:
        str: The image URI.
    """
    image_uri = image_uris.retrieve(
        framework=framework,
        region=region,
        version=version
    )
    return image_uri


if __name__ == "__main__":
    session = boto3.session.Session()
    current_region = session.region_name

    parser = argparse.ArgumentParser(description="Retrieve the SageMaker image URI for a given framework and model")
    parser.add_argument("--framework", type=str, default="huggingface-vllm-neuronx", help="The framework name")
    parser.add_argument("--region", type=str, default="us-east-1" if current_region is None else current_region, help="The AWS region")
    parser.add_argument("--version", type=str, default=None, help="The version of the framework")
    args = parser.parse_args()

    uri = retrieve_image_uri(
        framework=args.framework,
        region=args.region,
        version=args.version
    )
    print(f"Image URI: {uri}")
