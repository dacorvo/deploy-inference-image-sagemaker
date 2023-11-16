import argparse
import boto3
import json


def invoke(endpoint, prompt):
    smr = boto3.client("sagemaker-runtime")

    # Generation parameters
    parameters = {
        "do_sample": True,
        "top_p": 0.9,
        "temperature": 0.8,
        "max_new_tokens": 64,
        "repetition_penalty": 1.03,
    }

    request = {"inputs": prompt, "parameters": parameters, "stream": False}
    resp = smr.invoke_endpoint(
        EndpointName=endpoint,
        Body=json.dumps(request),
        ContentType="application/json",
    )
    print(resp["Body"].read())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Invoke a TGI service deployed on a sagemaker endpoint")
    parser.add_argument("endpoint", type=str, help="The endpoint invocation URL")
    parser.add_argument("--prompt", type=str, default="What is deep-learning ?", help="The test prompt")
    args = parser.parse_args()
    invoke(args.endpoint, args.prompt)
