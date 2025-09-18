import io
import json
import logging
import time
import random

import boto3
from locust.contrib.fasthttp import FastHttpUser

from locust import task, events

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@events.init_command_line_parser.add_listener
def _(parser):
    parser.add_argument("--region",
                        type=str,
                        env_var="AWS_DEFAULT_REGION",
                        default="us-east-1", help="The endpoint region")
    parser.add_argument("--prompt-file",
                        type=str,
                        default="alice.txt", help="The file containing the source for the prompt")
    parser.add_argument("--average-prompt-lines",
                        type=int,
                        default=2, help="The average number of lines to read for each prompt")
    parser.add_argument("--average-output-tokens",
                        type=int,
                        default=64, help="The average number of output tokens to generate")


@events.test_start.add_listener
def _(environment, **kw):
    with open(environment.parsed_options.prompt_file, "r") as f:
        environment.prompt_lines = f.readlines()


# Helper for reading lines from a stream
class JSONPayloadLineIterator:
    """
    A helper class for parsing a byte stream JSON output.

    While usually each PayloadPart event from the event stream will contain a byte array
    with a full json, this is not guaranteed and some of the json objects may be split across
    PayloadPart events. For example:
    ```
    {'PayloadPart': {'Bytes': b'{"data": '}}
    {'PayloadPart': {'Bytes': b'[" {"id""]}\n'}}
    ```

    This class accounts for this by concatenating bytes written via the 'write' function
    and then exposing a method which will return lines (ending with a '\n' character) within
    the buffer via the 'scan_lines' function. It maintains the position of the last read
    position to ensure that previous bytes are not exposed again.
    """

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
            if line and line[-1] == ord('\n'):
                self.read_pos += len(line)
                return line[:-1]
            try:
                chunk = next(self.byte_iterator)
            except StopIteration:
                if self.read_pos < self.buffer.getbuffer().nbytes:
                    continue
                raise
            if 'PayloadPart' not in chunk:
                print('Unknown event type:' + chunk)
                continue
            self.buffer.seek(0, io.SEEK_END)
            self.buffer.write(chunk['PayloadPart']['Bytes'])


class BotoClient:
    def __init__(self, endpoint_name, region_name):
        self.sagemaker_client = boto3.client("sagemaker-runtime", region_name=region_name)
        self.endpoint_name = endpoint_name

    def send(self, prompt, output_tokens):

        start_perf_counter = time.perf_counter()

        messages = [
            {
                "role": "system",
                "content": "Speak in a Medieval British style.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        body = {
            "messages": messages,
            "repetition_penalty": 1.0,
            "temperature": 0.5,
            "max_tokens": output_tokens,
            "stream": True,
            "stream_options": {
                "include_usage": True
            }
        }

        content = ""
        encoding_time = None
        prompt_tokens = 0
        completion_tokens = 0
        error = None
        try:
            response = self.sagemaker_client.invoke_endpoint_with_response_stream(
                EndpointName=self.endpoint_name,
                Body=json.dumps(body),
                ContentType="application/json",
            )
            event_stream = response["Body"]
            for line in JSONPayloadLineIterator(event_stream):
                line = line.decode("utf-8")
                if line.startswith("data:"):
                    # Cleanup line
                    line = line.lstrip("data:").rstrip("/n")
                    if line == " [DONE]":
                        break
                    response_data = json.loads(line)
                    choices = response_data["choices"]
                    if len(choices) > 0:
                        # This payload contains a chunk
                        chunk = response_data["choices"][0]["delta"]["content"]
                        content += chunk
                        if encoding_time is None:
                            # If this is the first chunk we receive, update encoding time
                            encoding_time = time.perf_counter() - start_perf_counter
                    elif "usage" in response_data:
                        usage = response_data["usage"]
                        prompt_tokens = usage["prompt_tokens"]
                        completion_tokens = usage["completion_tokens"]
        except Exception as e:
            logger.error(e)
            error = e

        total_time = time.perf_counter() - start_perf_counter
        events.request.fire(
            request_type="POST",
            name="total_time",
            response_time=total_time * 1000,
            response_length=prompt_tokens + completion_tokens,
            response=content,

            error=error
        )
        if encoding_time is not None:
            events.request.fire(
                request_type="POST",
                name="encoding_time",
                response_time=encoding_time * 1000,
                response_length=prompt_tokens,
            )
            events.request.fire(
                request_type="POST",
                name="decoding_time",
                response_time=(total_time - encoding_time) * 1000,
                response_length=completion_tokens,
            )



class BotoUser(FastHttpUser):
    abstract = True

    def __init__(self, env):
        super().__init__(env)
        self.client = BotoClient(self.host, self.environment.parsed_options.region)


class MyUser(BotoUser):
    @task
    def send_request(self):
        generator = random.Random()
        def randomize(average, variance):
            return max(1, int(generator.gauss(average, variance)))
        # Randomize the number of prompt lines
        average_num_lines = self.environment.parsed_options.average_prompt_lines
        variance_num_lines = average_num_lines * 0.1
        num_lines = randomize(average_num_lines, variance_num_lines)
        # Randomize the number of output tokens
        average_output_tokens = self.environment.parsed_options.average_output_tokens
        variance_output_tokens = average_output_tokens * 0.1
        output_tokens = randomize(average_output_tokens, variance_output_tokens)
        prompt = "\n".join(self.environment.prompt_lines[:num_lines])
        self.client.send(prompt, output_tokens)
