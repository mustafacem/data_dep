from openai import OpenAI
from langchain_openai import ChatOpenAI

model_for_summerization = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")