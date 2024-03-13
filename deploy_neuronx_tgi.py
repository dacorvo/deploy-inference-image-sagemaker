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
        "--auto_cast_type", type=str, default="fp16", choices=["fp32", "fp16", "bf16"], help="One of fp32, fp16, bf16."
    )
    parser.add_argument("--iam_role", default="sagemaker_execution_role", type=str)
    parser.add_argument("--region", default="us-east-1", type=str)
    parser.add_argument("--token", type=str, help="The HuggingGace token to use to fetch the model if gated or private.")
    args = parser.parse_args()

    os.environ['AWS_DEFAULT_REGION'] = args.region

    llm_image = args.llm_image
    if not "amazonaws.com" in llm_image:
        llm_image = get_huggingface_llm_image_uri("huggingface-neuronx",
                                                  version=llm_image)

    # TGI config
    def get_neuronx_tgi_config(model_id, batch_size, sequence_length, auto_cast_type, num_cores):

        max_input_length = sequence_length // 2
        max_total_tokens = sequence_length
        max_batch_prefill_tokens = batch_size * max_input_length
        max_batch_total_tokens = batch_size * sequence_length

        return {
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

    tgi_config = get_neuronx_tgi_config(args.model_id,
                                        args.batch_size,
                                        args.sequence_length,
                                        args.auto_cast_type,
                                        args.num_cores)
    if args.token:
        tgi_config["HUGGING_FACE_HUB_TOKEN"] = args.token

    deploy_image(llm_image,
                 tgi_config,
                 instance_type=args.instance_type,
                 iam_role=args.iam_role)
