# Sagemaker Hugging Face service deployment scripts

## Prerequisites

You need an existing Sagemaker-compatible image, possibly created from: https://github.com/awslabs/llm-hosting-container.

You also need to install the requirements:

```shell
pip install -r requirements.txt
```

Before running any script you need to export your AWS CLI credentials, namely:

```shell
export AWS_ACCESS_KEY_ID=<ACCESS_KEY_ID>
export AWS_SECRET_ACCESS_KEY=<SECRET_ACCESS_KEY>
export AWS_SESSION_TOKEN=<SESSION_TOKEN>
```

And you need also to set a default region:

```shell
export AWS_DEFAULT_REGION=<REGION>
```

## Push an image to ECR

```shell
./push_ecr_image.sh <IMAGE>:<TAG> <ECR_REPOSITORY> <REGION>
```

## Deploy a Sagemaker service

```shell
python deploy_image.py \
    --llm_image <USER>.dkr.ecr.<REGION>.amazonaws.com/<ECR_REPOSITORY:<TAG> \
    --model_id  <HF_MODEL_ID> \
    --instance_type ml.<INSTANCE_TYPE> \
    --region <REGION>
```

Note: you can specify the exact deployment configuration by passing some arguments
to deploy_image.py (see `python deploy_image.py --help` for the exact list). Otherwise a default configuration will be selected.

## Test the endpoint

```shell
export <SAGEMAKER_ENDPOINT_NAME>
python invoke_endpoint.py <SAGEMAKER_ENDPOINT_NAME> --max_new_tokens 128
```

## Benchmark the endpoint

```shell
./benchmark/benchmark.sh <SAGEMAKER_ENDPOINT_NAME> \
                         <CONCURRENT_USERS> \
                         <DURATION> \
                         <AVG_PROMPT_LINES> \
                         <AVG_OUTPUT_TOKENS>
```

The benchmark results csv files prefixed with <SAGEMAKER_ENDPOINT_NAME> will be generated in the current directory.
Look for the summary file for the Time-to-first-token and output tokens Throughput.
