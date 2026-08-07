from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from huggingface_hub.errors import HfHubHTTPError
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, load_prompt
from langchain_core.output_parsers import JsonOutputParser
import os

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

parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me the name, age and city of a fictional person. \n {format_instruction}.",
    input_variables=[],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)


chain = template | model | parser
final_res = chain.invoke({})

print("Final Result:\n", final_res)