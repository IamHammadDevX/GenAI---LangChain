from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os

llm = HuggingFacePipeline.from_model_id(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 100, "temperature": 0.7},
)

model = ChatHuggingFace(llm=llm)
result = model.invoke("What is the name of capital of France?")
print(result.content)