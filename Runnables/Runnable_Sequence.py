from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
import os

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError(
        "OPENROUTER_API_KEY is missing from the .env file."
    )

model = ChatOpenAI(
    model="openrouter/free",
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

prompt = PromptTemplate(template="Write a joke on {topic}", input_variables=["topic"])

prompt1 = PromptTemplate(template="Explain the Following Joke - {text}", input_variables=["text"])

parser = StrOutputParser()

chain = RunnableSequence(prompt, model, parser, prompt1, model, parser)

res = chain.invoke({"topic": "AI"})
print(res)