import asyncio
import logging

from model import config
from model.main import DocumentReader


if __name__ == "__main__":

    # load context document
    with open("numpy-ref.pdf", "rb") as f:
        f.read()

        doc_reader = DocumentReader(
            azure_openai_endpoint=config.endpt,
            azure_openai_key=config.key,
            azure_openai_deployment=config.deployment,
            azure_openai_deployment_version=config.deployment_version
        )

    # utilize data and call AzureOpenAICaller methods usign azopenai object
    response_docreader = asyncio.run(
        doc_reader.query_llm("what are the contents of this book?")
    )
    logging.info(f"{response_docreader=}")
