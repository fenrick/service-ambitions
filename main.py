import getpass
import os
import json
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.utils.json import (parse_json_markdown)
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check if the environment variable is set, if not prompt for it
if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

# Load the system prompt from a markdown file
try:
  with open("prompt.md",'r', encoding='utf-8') as file:
    system_prompt = file.read()
except FileNotFoundError:
  print("Prompt file not found. Please create a prompt.md file in the current directory.")
  exit(1)
except Exception as e:
  print(f"An error occurred while reading the prompt file: {e}")
  exit(1)

# Load the services from a JSON file
try:
  with open("sample-services.json", 'r', encoding='utf-8') as file:
    services = json.load(file)
except FileNotFoundError:
  print("Services file not found. Please create a services.json file in the current directory.")
  exit(1)
except Exception as e:
  print(f"An error occurred while reading the services file: {e}")
  exit(1)

# Initialize the chat model
prompt_template = ChatPromptTemplate([
  ("system", "{system_prompt}"),
  ("user", "{user_prompt}"),
])

# Initialize the chat model with the specified parameters
model = init_chat_model(model="o4-mini", model_provider="openai")

# Process each service
for i, service in enumerate(services, start=1):
  print(f"Processing service {i}: {service['name']}")

  # Convert the service details to a JSON string
  service_details = json.dumps(service)

  # Create the prompt using the template and the system prompt
  prompt = prompt_template.invoke({"system_prompt": system_prompt, "user_prompt": service_details})

  # Invoke the model with the prompt
  response = model.invoke(prompt)

  # Parse the response a json parser
  parsed_response = parse_json_markdown(response.content)

  response_string = json.dumps(parsed_response)

  # Output the response in to ambitions.json
  with open("ambitions.json", 'a', encoding='utf-8') as file:
    # Write the service name and response to the file
    file.write(f"{response_string}\n")
