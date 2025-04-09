import logging

from model import config
from model.call_openai import AzureOpenAICaller


if __name__ == "__main__":

    azopenai = AzureOpenAICaller(
        azure_openai_endpoint=config.endpt,
        azure_openai_key=config.key,
        azure_openai_deployment=config.deployment,
        azure_openai_deployment_version=config.deployment_version
    )

    # utilize data and call AzureOpenAICaller methods usign azopenai object
    response_openai = azopenai.call_openai_chat(
        input_text="Explain in detail what are transformers.",
        system_prompt="You are an AI/ML expert."
    )
    logging.info(f"{response_openai=}")
