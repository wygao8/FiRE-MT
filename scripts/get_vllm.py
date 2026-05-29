import re
import time
from vllm import LLM, SamplingParams

MODEL_NAME = "QwQ-32B"
MODEL_PATH = ""


class VllmModel:
    def __init__(self, model_name = MODEL_NAME, model_path=MODEL_PATH, tensor_parallel_size=2, gpu_memory_utilization=0.95, max_model_len=8192):
        print(f"tensor parallel size = {tensor_parallel_size}")
        self.model_name = model_name
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            print(f"INFO: Loading {self.model_name} by vLLM, tensor parallel size = {self.tensor_parallel_size}, gpu memory utilization = {self.gpu_memory_utilization}")
            self._llm = LLM(
                model=self.model_path,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                enable_chunked_prefill=True
            )
        return self._llm

    @staticmethod
    def parse_thinking_content(prompt, response):
        if "</think>" not in response:
            print("WARNING no </think> tag in response")
            return {"content": response, "thinking_content": None}

        parts = response.split("</think>", 1)
        message = {}
        if len(parts) == 2:
            reasoning_raw, content = parts
            message["content"] = content

            reasoning = reasoning_raw.strip()
            message["reasoning_content"] = reasoning if reasoning else None
        else:
            print("WARNING: </think> not found")
            message["content"] = response
            message["reasoning_content"] = None
        return message

    def get_response(self, messages, temperature=0.6):
        sampling_params = SamplingParams(
            temperature=temperature, 
            top_p=0.95,
            top_k=20,
            max_tokens=3000
        )
        outputs = self.llm.chat(
            messages, 
            sampling_params,
            chat_template_kwargs={"enable_thinking": True},  # Set to False to strictly disable thinking
        )
        response = [VllmModel.parse_thinking_content(output.prompt, output.outputs[0].text) for output in outputs]
        return response