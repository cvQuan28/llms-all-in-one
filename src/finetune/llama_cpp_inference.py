from transformers import (AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, pipeline)
from langchain.llms.llamacpp import LlamaCpp
import os
import torch
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from src.finetune.inference import Inference


class LlamaCppInference(Inference):
    def __init__(self, model_path, max_new_tokens=256, temperature=0.7, top_p=0.95, top_k=1, repetition_penalty=1.15,
                 n_gpu_layers=-1, n_ctx=4048, verbose=False, n_batch=512):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.prefix1 = ""
        self.prefix2 = ""
        self.model = None
        self.model_chain = None
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        self.n_ctx = n_ctx
        self.n_batch = n_batch

    def load_model(self):
        load_model_status = 0
        msg = None
        try:
            self.model = LlamaCpp(model_path=self.model_path, n_gpu_layers=self.n_gpu_layers, n_ctx=self.n_ctx,
                                  max_tokens=self.max_new_tokens, temperature=self.temperature,
                                  n_batch=self.n_batch,
                                  verbose=self.verbose, top_k=self.top_k, top_p=self.top_p,
                                  repeat_penalty=self.repetition_penalty)

            template = """<s>[INST] <<SYS>>
Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực. Hãy luôn trả lời một cách hữu ích nhất có thể, đồng thời giữ an toàn.
Câu trả lời của bạn không nên chứa bất kỳ nội dung gây hại, phân biệt chủng tộc, phân biệt giới tính, độc hại, nguy hiểm hoặc bất hợp pháp nào. Hãy đảm bảo rằng các câu trả lời của bạn không có thiên kiến xã hội và mang tính tích cực.Nếu một câu hỏi không có ý nghĩa hoặc không hợp lý về mặt thông tin, hãy giải thích tại sao thay vì trả lời một điều gì đó không chính xác. Nếu bạn không biết câu trả lời cho một câu hỏi, hãy trẳ lời là bạn không biết và vui lòng không chia sẻ thông tin sai lệch.
<</SYS>>

{prompt} [/INST]
"""
            prompt = PromptTemplate.from_template(template)
            self.model_chain = LLMChain(prompt=prompt, llm=self.model)
        except Exception as e:
            load_model_status = -1
            msg = e
        return load_model_status, msg

    def infer(self, input):
        return self.model(input)

    def run(self, input):
        return self.model_chain.run(input)

    def free_memory(self):
        if self.model:
            del self.model
            self.model = None
