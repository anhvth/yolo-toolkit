# Define the URL where Label Studio is accessible
LABEL_STUDIO_URL = 'http://localhost:8080'
# API key is available at the Account & Settings page in Label Studio UI
from dotenv import load_dotenv
import os
load_dotenv()
LABEL_STUDIO_API_KEY = os.getenv("LABEL_STUDIO_API_KEY")
print(f"LABEL_STUDIO_API_KEY: {LABEL_STUDIO_API_KEY}")

# Import the SDK and the client module
from label_studio_sdk import LabelStudio

# Connect to the Label Studio API 
client = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=LABEL_STUDIO_API_KEY)

# A basic request to verify connection is working
me = client.users.whoami()

print("username:", me.username)
print("email:", me.email)
