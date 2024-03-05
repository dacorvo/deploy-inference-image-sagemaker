# Sagemaker Hugging Face TGI service deployment scripts

## Prerequisites

You need an existing Sagemaker HF TGI image, possibly created from: https://github.com/awslabs/llm-hosting-container.

You also need to export your AWS CLI credentials, namely:

```
export AWS_ACCESS_KEY_ID=<ACCESS_KEY_ID>
export AWS_SECRET_ACCESS_KEY=<SECRET_ACCESS_KEY>
export AWS_SESSION_TOKEN=<SESSIOn_TOKEN>
```

And finally you need to set a default region:

```
export AWS_DEFAULT_REGION=<REGION>
```

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

