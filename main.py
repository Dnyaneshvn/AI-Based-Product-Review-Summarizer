#!pip install -q transformers einops accelerate langchain bitsandbytes sentencepiece accelerate langchain_community

#!huggingface-cli login

from langchain import HuggingFacePipeline, LLMChain
from transformers import AutoTokenizer
import transformers
import torch

model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = transformers.pipeline("text-generation",
                                 model=model, tokenizer=tokenizer, torch_dtype=torch.bfloat16,trust_remote_code=True, device_map="auto", max_length = 1000 ,
                                 do_sample=True, top_k=10,num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)

llm = HuggingFacePipeline (pipeline = pipeline, model_kwargs= {'temperature':0})

from langchain import PromptTemplate

template = """
You have to generate a concise summary in english with grammatically improvements via analyzing reviews and Follow below guidelines:
1.Introduction: Begin with a brief overview of the product and purpose.
2.Main points: Summarize the main features of the product ensuring to cover all key aspects.
3.Semantic analysis: Analyze the reviews as positive, negative, or neutral about the product.
4.Conclusion: Conclude the summary with a final statement that summarizes the product's key message.

Summary: {text}
"""

prompt = PromptTemplate(template=template, input_variables=["text"])

llm_chain = LLMChain(prompt = prompt, llm = llm)

import pandas as pd
df = pd.read_csv('path')
reviews = df['column_index'].tolist()

batch_size = 2
for i in range(0, len(reviews), batch_size):
    batch_reviews = reviews[i:i+batch_size]

text = " ".join(batch_reviews)
summary = llm_chain.run(text)
print(summary)
