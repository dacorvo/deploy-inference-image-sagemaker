import argparse
import boto3
import io
import json


# Helper for reading lines from a stream
class LineIterator:
    def __init__(self, stream):
        self.byte_iterator = iter(stream)
        self.buffer = io.BytesIO()
        self.read_pos = 0

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            self.buffer.seek(self.read_pos)
            line = self.buffer.readline()
            if line and line[-1] == ord("\n"):
                self.read_pos += len(line)
                return line[:-1]
            try:
                chunk = next(self.byte_iterator)
            except StopIteration:
                if self.read_pos < self.buffer.getbuffer().nbytes:
                    continue
                raise
            if "PayloadPart" not in chunk:
                print("Unknown event type:" + chunk)
                continue
            self.buffer.seek(0, io.SEEK_END)
            self.buffer.write(chunk["PayloadPart"]["Bytes"])


def invoke(endpoint, prompt, stream=False):
    smr = boto3.client("sagemaker-runtime")

    # Generation parameters
    parameters = {
        "do_sample": True,
        "top_p": 0.9,
        "temperature": 0.8,
        "max_new_tokens": 64,
        "repetition_penalty": 1.03,
    }

    request = {"inputs": prompt, "parameters": parameters, "stream": stream}
    if stream:
        resp = smr.invoke_endpoint_with_response_stream(
            EndpointName=endpoint,
            Body=json.dumps(request),
            ContentType="application/json",
        )
        def response_reader(resp):
            output = ""
            for c in LineIterator(resp["Body"]):
                c = c.decode("utf-8")
                if c.startswith("data:"):
                    chunk = json.loads(c.lstrip("data:").rstrip("/n"))
                    if chunk["token"]["special"]:
                        continue
                    output += chunk["token"]["text"]
                yield output
        for t in response_reader(resp):
            print(t, end='\r')
    else:
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
    parser.add_argument("--stream", action='store_true', help="Stream generated tokens")
    args = parser.parse_args()
    invoke(args.endpoint, args.prompt, args.stream)
