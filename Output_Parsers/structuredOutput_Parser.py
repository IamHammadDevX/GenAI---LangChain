from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from huggingface_hub.errors import HfHubHTTPError
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, load_prompt
from langchain.output_parsers import StructuredOutputParser, ResponseSchema # type: ignore
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


schema = [
    ResponseSchema(name="name", description="The name of the person"),
    ResponseSchema(name="age", description="The age of the person"),
    ResponseSchema(name="city", description="The city where the person lives")
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="Give me 3 facts about {topic}. \n {format_instruction}.",
    input_variables=["topic"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({"topic": "Fictional Person"})

print(result)