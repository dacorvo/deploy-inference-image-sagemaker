#!/bin/sh
LLM_IMAGE="754289655784.dkr.ecr.us-east-2.amazonaws.com/neuronx-tgi:1.0.2-0.0.14dev"
MODEL_ID="aws-neuron/gpt2-seqlen-1024-bs-16"
INSTANCE_TYPE="ml.inf2.xlarge"
REGION="us-east-2"

python deploy_neuronx_tgi.py \
    --llm_image ${LLM_IMAGE} \
    --model_id ${MODEL_ID} \
    --instance_type ${INSTANCE_TYPE} \
    --region ${REGION}
