import boto3
import gradio as gr
import io
import json
import os
from transformers import AutoTokenizer


aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID", None)
aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY", None)
aws_session_token = os.environ.get("AWS_SESSION_TOKEN", None)
region = os.environ.get("AWS_REGION", None)
endpoint_name = os.environ.get("SAGEMAKER_ENDPOINT_NAME", None)

if (
    aws_access_key_id is None
    or aws_secret_access_key is None
    or region is None
    or endpoint_name is None
):
    raise Exception(
        "Please set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION and SAGEMAKER_ENDPOINT_NAME environment variables"
    )

boto_session = boto3.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    aws_session_token=aws_session_token,
    region_name=region,
)

smr = boto_session.client("sagemaker-runtime")

# We need the LLama tokenizer for chat templates
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tokenizer.use_default_system_prompt = False


def format_chat_prompt(message, history, max_tokens):
    # First, convert history to a chat
    chat = []
    for interaction in history:
        chat.append({"role": "user", "content" : interaction[0]})
        chat.append({"role": "assistant", "content" : interaction[1]})
    chat.append({"role": "user", "content" : message})
    for i in range(0, len(chat), 2):
        # Generate candidate prompt with the last n-i entries
        prompt = tokenizer.apply_chat_template(chat[i:], tokenize=False)
        # Tokenize to check if we're over the limit
        tokens = tokenizer(prompt)
        if len(tokens.input_ids) <= max_tokens:
            # We're good, stop here
            return prompt
    # We shall never reach this line
    raise SystemError

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


# query client using streaming mode
def generate(message, history):
    # Convert history to a chat prompt
    prompt = format_chat_prompt(message, history, max_tokens=2048)

    # Request generation parameters
    parameters = {
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.9,
        "temperature": 0.9,
        "max_new_tokens": 1024,
        "repetition_penalty": 1.2,
    }
    request = {"inputs": prompt, "parameters": parameters, "stream": True}
    resp = smr.invoke_endpoint_with_response_stream(
        EndpointName=endpoint_name,
        Body=json.dumps(request),
        ContentType="application/json",
    )

    # Process streamed response
    text = ""
    for c in LineIterator(resp["Body"]):
        c = c.decode("utf-8")
        if c.startswith("data:"):
            chunk = json.loads(c.lstrip("data:").rstrip("/n"))
            if chunk["token"]["special"]:
                continue
            text += chunk["token"]["text"]
            yield text
    return text

markdown_header = """
            <div style="text-align: center; max-width: 650px; margin: 0 auto; display:grid; gap:25px;">
                <img class="logo" src="https://huggingface.co/datasets/philschmid/assets/resolve/main/aws-neuron_hf.png" alt="Hugging Face Logo"
                    style="margin: auto; max-width: 14rem;">
                <h1 style="font-weight: 900; margin-bottom: 7px;margin-top:5px">
                Chat with LLama on AWS INF2 ⚡
                </h1>
            <p style="margin-bottom: 10px; font-size: 94%; line-height: 23px;">
                This demo is running on <a style="text-decoration: underline;" href="https://aws.amazon.com/ec2/instance-types/inf2/?nc1=h_ls">AWS Inferentia2</a>,
                to achieve efficient and cost-effective inference.
                <a href="https://huggingface.co/blog/inferentia-llama2" target="_blank">How does it work?</a>
            </p>
            </div>
            <div style="text-align: center; max-width: 650px; margin: 0 auto; display:grid; gap:25px;">
            <p style="margin-bottom: 10px; font-size: 94%; line-height: 23px;">
            The chat context is limited to 2048 tokens, and older interactions are progressively 'forgotten'.
            Use the clear button to restart a new conversation.
            </p>
            </div>
"""

gr.ChatInterface(
    generate,
    chatbot=gr.Chatbot(),
    textbox=gr.Textbox(placeholder="Chat with me!", container=False, scale=7),
    description=markdown_header,
    theme="abidlabs/Lime",
    examples=[
        "My favorite color is blue, and my favorite fruit is strawberry.",
        "What is the color of my favorite fruit ?",
        "Name a fruit that is on my favorite color.",
    ],
    cache_examples=False,
    retry_btn="Retry",
    undo_btn="Undo",
    clear_btn="Clear").queue().launch()
