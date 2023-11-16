import os
from sagemaker.huggingface import get_huggingface_llm_image_uri
import argparse

from transformers import AutoConfig

from deploy_image import deploy_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy a service based on an HuggingFace image on sagemaker")
    parser.add_argument("--llm_image", type=str, required=True, help="Either a full image URI or just the tag for an HuggingFace image")
    parser.add_argument("--model_id", type=str, required=True, help="The HuggingFace model id")
    parser.add_argument("--instance_type", type=str, required=True, help="The Sagemaker instance type")
    parser.add_argument("--iam_role", default="sagemaker_execution_role", type=str)
    parser.add_argument("--region", default="us-east-1", type=str)
    parser.add_argument("--token", type=str, help="The HuggingGace token to use to fetch the model if gated or private.")
    args = parser.parse_args()

    os.environ['AWS_DEFAULT_REGION'] = args.region

    llm_image = args.llm_image
    if not "amazonaws.com" in llm_image:
        llm_image = get_huggingface_llm_image_uri("huggingface", version=llm_image)

    # TGI config
    def get_neuronx_tgi_config(model_id):

        config = AutoConfig.from_pretrained(model_id)
        batch_size = config.neuron["batch_size"]
        sequence_length = config.neuron["sequence_length"]
        max_concurrent_requests = batch_size
        max_input_length = sequence_length // 2
        max_total_tokens = sequence_length
        max_batch_prefill_tokens = batch_size * max_input_length
        max_batch_total_tokens = batch_size * sequence_length

        return {
            "HF_MODEL_ID": model_id,
            "MAX_CONCURRENT_REQUESTS": f"{max_concurrent_requests}",
            "MAX_INPUT_LENGTH": f"{max_input_length}",
            "MAX_TOTAL_TOKENS": f"{max_total_tokens}",
            "MAX_BATCH_PREFILL_TOKENS": f"{max_batch_prefill_tokens}",
            "MAX_BATCH_TOTAL_TOKENS": f"{max_batch_total_tokens}",
        }

    tgi_config = get_neuronx_tgi_config(args.model_id)
    if args.token:
        tgi_config["HUGGING_FACE_HUB_TOKEN"] = args.token

    deploy_image(llm_image,
                 tgi_config,
                 instance_type=args.instance_type,
                 iam_role=args.iam_role)
