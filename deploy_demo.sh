#!/bin/sh
LLM_IMAGE="754289655784.dkr.ecr.us-east-2.amazonaws.com/neuronx-tgi:1.0.2-0.0.14dev"
MODEL_ID="aws-neuron/Llama-2-7b-chat-hf-seqlen-2048-bs-4"
INSTANCE_TYPE="ml.inf2.8xlarge"
REGION="us-east-2"

python deploy_neuronx_tgi.py \
    --llm_image ${LLM_IMAGE} \
    --model_id ${MODEL_ID} \
    --instance_type ${INSTANCE_TYPE} \
    --region ${REGION}
