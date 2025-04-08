import os


api_auth_name = "azure_openai_fastapi_auth_key"
API_AUTH_KEY = "123456789"

# azure openai resource configs
endpt = os.getenv("azure_openai_endpt")
key = os.getenv("azure_openai_key")
deployment = os.getenv("azure_openai_deployment")
deployment_version = os.getenv("azure_openai_deployment_version")
