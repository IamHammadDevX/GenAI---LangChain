from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal, Optional
from pydantic import BaseModel, Field
import os

# from Structured_outputs.with_structured_output_typeddict import Review

load_dotenv()
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not token:
    raise ValueError(
        "HUGGINGFACEHUB_API_TOKEN is missing from the .env file."
    )

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    huggingfacehub_api_token=token,
    max_new_tokens=4096, 
    temperature=0.1
)

model = ChatHuggingFace(llm=llm)

# schema
Review_schema = {
    "title": "Review",
    "type": "object",
    "properties": {
        "key_themes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "A list of key themes mentioned in the review in a list"
        },
        "summary": {
            "type": "string",
            "description": "A brief summary of the review"
        },
        "sentiment": {
            "type": "string",
            "enum": ["pos", "neg"],
            "description": "The sentiment of the review, Either 'positive', 'negative'"
        },
        "pros": {
            "type": "array",
            "items": {"type": "string"},
            "description": "A list of pros mentioned in the review in a list"
        },
        "cons": {
            "type": "array",
            "items": {"type": "string"},
            "description": "A list of cons mentioned in the review in a list"
        },
        "name": {
            "type": "string",
            "description": "The name of the product being reviewed"
        }
    },
    "required": ["key_themes", "summary", "sentiment"]
}

structured_output = model.with_structured_output(
    Review_schema,
    method="json_schema"
)

result = structured_output.invoke(
    """
    I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it's an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I'm gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung's One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.Pros:Insanely powerful processor (great for gaming and productivity)Stunning 200MP camera with incredible zoom capabilitiesLong battery life with fast chargingS-Pen support is unique and usefulCons:Bulky and heavy—not great for one-handed useBloatware still exists in One UIExpensive compared to competitors
    
    Review by John
    """
)

print(result)