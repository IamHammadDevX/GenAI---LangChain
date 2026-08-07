from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from huggingface_hub.errors import HfHubHTTPError
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, load_prompt
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import os

from typer import prompt

load_dotenv()
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not token:
    raise ValueError(
        "HUGGINGFACEHUB_API_TOKEN is missing from the .env file."
    )

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=token
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(..., description="The name of the person")
    age: int = Field(gt=18, description="The age of the person")
    city: str = Field(..., description="The city where the person lives")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate the name, age and city of a fictional {place} person. \n {format_instruction}.",
    input_variables=["place"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({"place": "American"})
print("Final Result:\n", result)