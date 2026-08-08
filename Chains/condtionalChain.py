from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Any, Dict, Literal
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

class FeedbackSentiment(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="The sentiment of the feedback, either positive or negative."
    )

prompt1 = PromptTemplate(
    template="Classify the sentiment of the following text as either positive or negative: \n{feedback} \n {format_instruction}",
    input_variables=["feedback"],
    partial_variables={"format_instruction": PydanticOutputParser(pydantic_object=FeedbackSentiment).get_format_instructions()}
)



classifier_chain = prompt1 | model | PydanticOutputParser(pydantic_object=FeedbackSentiment)

prompt2 = PromptTemplate(
    template="Generate a response to the following positive feedback: \n{feedback}",
    input_variables=["feedback"]
)

prompt3 = PromptTemplate(
    template="Generate a response to the following negative feedback: \n{feedback}",
    input_variables=["feedback"]
)

branch_chain = RunnableBranch(
   (lambda x: x.sentiment == "positive", prompt2 | model | StrOutputParser()),
   (lambda x: x.sentiment == "negative", prompt3 | model | StrOutputParser()),
   RunnableLambda(lambda x: "No feedback provided.")
)

chain = classifier_chain | branch_chain
res = chain.invoke({"feedback": "The product is amazing and exceeded my expectations!"})
print("Final Result:\n", res)

chain.get_graph().print_ascii()