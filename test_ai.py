# Import prerequisite libraries
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Setting the API key
# Define the user prompt message
prompt = "Hello!"
# Create a chatbot using ChatCompletion.create() function
completion = client.chat.completions.create(# Use GPT 3.5 as the LLM
model="gpt-3.5-turbo",
# Pre-define conversation messages for the possible roles 
messages=[
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": prompt}
])
# Print the returned output from the LLM model
print(completion.choices[0].message)