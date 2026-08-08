from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
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

model2 = ChatOpenAI(
    model="openrouter/free",
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text.\n{text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template="Generate 5 short question answers from the following topic:\n {text}.",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into single document \n notes -> {notes} and quiz -> {quiz}.",
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

ParallelChain = RunnableParallel({
    "notes": prompt1 | model | parser,
    "quiz": prompt2 | model2 | parser
})

merge_chain = prompt3 | model | parser

chain = ParallelChain | merge_chain

text = """
Transformer architecture is a deep learning model introduced in 2017 in the paper **“Attention Is All You Need.”** It is widely used in modern language models such as GPT.

* **Input Embedding:** Converts words/tokens into numerical vectors.
* **Positional Encoding:** Adds information about the position/order of tokens.
* **Self-Attention:** Helps the model understand relationships between different words in the input.
* **Multi-Head Attention:** Uses multiple attention mechanisms to learn different relationships simultaneously.
* **Feed-Forward Network:** Further processes the information learned by attention.
* **Encoder & Decoder:** The original Transformer contains an encoder for understanding input and a decoder for generating output.
* **Output Layer:** Produces probabilities for the next token or final prediction.

**In simple words:** A Transformer understands which words are important to each other using **attention**, allowing it to process text efficiently and capture long-range relationships.

**Flow:**

```text
Input Text
   ↓
Tokenization
   ↓
Embedding + Positional Encoding
   ↓
Self-Attention
   ↓
Feed-Forward Network
   ↓
Transformer Layers
   ↓
Output
```

GPT models mainly use the **decoder part** of the Transformer architecture.

"""

result = chain.invoke({
    "text": text
})

print("Final Result:\n", result)

chain.get_graph().print_ascii()