import os
import openai
import gradio as gr


client = openai.OpenAI(
    api_key="EMPTY",
    base_url=os.environ.get("VLLM_ENDPOINT", "http://localhost:8000/v1"),
)


def predict(message, history):
    history_openai_format = [
        {"role": "system", "content": "You are a helpful assistent. Answer in markdown format. Be concise."}]
    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human})
        history_openai_format.append({"role": "assistant", "content": assistant})
    history_openai_format.append({"role": "user", "content": message})

    default_model = client.with_options(max_retries=3, timeout=10).models.list().data[0].id
    stream = client.with_options(max_retries=3).chat.completions.create(
        messages=history_openai_format,
        model=default_model,
        stream=True,
    )

    partial_message = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            partial_message = partial_message + chunk.choices[0].delta.content
            yield partial_message


gr.ChatInterface(
    predict,
    title="LLM Inference Engine",
    description="Efficient LLM Inferencing on a NVIDIA L4 Core GPU with 24 GiB of VRAM.",
    examples=[
        "Tell me about yourself.",
        "Tell me a joke.",
        "Tell me about CIS 565 at UPenn.",
        "Who are the instructors of CIS 565 at UPenn?",
        "Show some of the student projects of CIS 565.",
        "What is attention mechanism in deep learning?",
        "Write a GEMM program in CUDA."
    ],
).queue().launch(server_name="0.0.0.0", server_port=7860)
