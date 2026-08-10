import random
from abc import ABC, abstractmethod

class RUNNABLE(ABC):

    @abstractmethod
    def invoke(input_data):
        pass


class LLMdemo(RUNNABLE):

    def __init__(self):
        print("LLM Created")
    
    def invoke(self, prompt):

        resonponse_list = [
            "Islamabad is the capital of Pakistan.",
            "IPL is the Frenchise of India",
            "ML stands for Machine learning"
        ]

        return {"response": random.choice(resonponse_list)}

    def predict(self, prompt):

        resonponse_list = [
            "Islamabad is the capital of Pakistan.",
            "IPL is the Frenchise of India",
            "ML stands for Machine learning"
        ]

        return {"response": random.choice(resonponse_list)}



class Promptdemo(RUNNABLE):

    def __init__(self, template, input_vars):
        self.template = template
        self.input_var = input_vars

    def invoke(self, input_dict):    
        return self.template.format(**input_dict)
    
    def format(self, input_dict):
        return self.template.format(**input_dict)

class fakeStrOutputParser(RUNNABLE):
    def invoke(self, input_data):
        return input_data["response"]

class RunnableConnector(RUNNABLE):
    def __init__(self, runnable_list):
        self.runnable_list = runnable_list
    def invoke(self, input_data):
        for runnable in self.runnable_list:
            input_data = runnable.invoke(input_data)
        return input_data


llm = LLMdemo()

prompt = Promptdemo(template="What is the capital of {country}?", input_vars=["country"])
parser = fakeStrOutputParser()
chan = RunnableConnector([prompt, llm, parser])
res = chan.invoke({"country": "Pakistan"})

print(res)