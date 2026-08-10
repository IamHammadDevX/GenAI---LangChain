import random

class LLMdemo():

    def __init__(self):
        print("LLM Created")
    
    def predict(self, prompt):

        resonponse_list = [
            "Islamabad is the capital of Pakistan.",
            "IPL is the Frenchise of India",
            "ML stands for Machine learning"
        ]

        return {"response": random.choice(resonponse_list)}

llm = LLMdemo()
res = llm.predict("What is the capital of Pakistan?")

class Promptdemo():

    def __init__(self, template, input_vars):
        self.template = template
        self.input_var = input_vars
    
    def format(self, input_dict):

        return self.template.format(**input_dict)

prompt = Promptdemo(template="What is the capital of {country}?", input_vars=["country"])
prmt = prompt.format({"country": "Pakistan"})


class FakeLLMChain():
    def __init__(self, prompt, llm):
        self.prompt = prompt
        self.llm = llm
    
    def run(self, input_dict):
        formatted_prompt = self.prompt.format(input_dict)
        result = self.llm.predict(formatted_prompt)
        return result["response"]


fakeChain = FakeLLMChain(prompt, llm)
result = fakeChain.run({"country": "Pakistan"})
print("Final Result from FakeLLMChain:\n", result)