from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
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

prompt1 = PromptTemplate(template="Generate a Tweet about {topic}", input_variables=["topic"])

prompt2 = PromptTemplate(template="Generate a LinkedIn post about {topic}", input_variables=["topic"])

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel(
    {
        "joke": RunnablePassthrough(),
        "explanation": RunnableSequence(prompt2, model, parser)
    }
)

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)
print(final_chain.invoke({"topic": "data science"}))