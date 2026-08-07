from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, load_prompt
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not token:
    raise ValueError(
        "HUGGINGFACEHUB_API_TOKEN is missing from the .env file."
    )

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    huggingfacehub_api_token=token
)

model = ChatHuggingFace(llm=llm)

# 1st prompt - detailed prompt
template1 = PromptTemplate(
    template="Write a detailed report on {topic}.",
    input_variables=["topic"]
)


# summary
template2 = PromptTemplate(
    template="Write a 5 line summary on following text.\n {text}",
    input_variables=["text"]
)

parser = StrOutputParser()
chain = template1 | model | parser | template2 | model | parser
result = chain.invoke({"topic": "The impact of climate change on global agriculture."})
print("Detailed Report:\n", result)
