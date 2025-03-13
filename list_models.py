import os
from openai import OpenAI

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

# Setting the API key

# Perform tasks using OpenAI API
models = client.models.list()  # List all OpenAI models
print(models)