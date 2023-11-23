# Sagemaker Hugging Face TGI service deployment scripts

## Prerequisite

You need an existing Sagemaker HF TGI image, possibly created from: https://github.com/awslabs/llm-hosting-container.

## Push a TGI image to ECR

```
./push_ecr_image.sh <IMAGE>:<TAG> <USER> <REGION>
```

## Deploy a TGI neuronx service

```
python deploy_neuronx_tgi.py \
    --llm_image <USER>.dkr.ecr.<REGION>.amazonaws.com/<IMAGE>:<TAG> \
    --model_id  <HF_MODEL_ID> \
    --instance_type ml.<INSTANCE_TYPE> \
    --region <REGION>
```

