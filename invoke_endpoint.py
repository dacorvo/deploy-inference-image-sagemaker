import argparse

from sagemaker.huggingface import HuggingFacePredictor


def invoke(endpoint,
           prompt="What is Deep Learning ?",
           max_new_tokens=20,
           top_k=50,
           top_p=0.9,
           temperature=1.0):

    predictor = HuggingFacePredictor(endpoint_name=endpoint)
    # send request
    output = predictor.predict(
        {
            "inputs": prompt,
            "parameters": {
                "do_sample": True,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
            }
        }
    )
    print(output[0]["generated_text"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("endpoint", type=str)
    parser.add_argument("--prompt", type=str, default="What is Deep-learning ?")
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()
    invoke(args.endpoint,
           prompt=args.prompt,
           max_new_tokens=args.max_new_tokens,
           top_k=args.top_k,
           top_p=args.top_p,
           temperature=args.temperature)


if __name__ == "__main__":
    main()

