from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter, TokenTextSplitter
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.llms.litellm import LiteLLM
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import json

from model.pdf_extractor import PDFText
from model import config

# llm settings, use whichever
llm = LiteLLM(
    model="azure/gpt-4o",
    api_base=config.azure_openai_api_endpoint,
    api_key=config.azure_openai_api_key,
    temperature=0.5,
    system_prompt="You read books and answer questions over it.",
    caching=True,
    set_verbose=False,
    timeout=600,
    response_format="json",
    max_retries=3
)
llm = AzureOpenAI(
    model="gpt-4o",
    temperature=0.5,
    api_key=config.azure_openai_api_key,
    api_version=config.azure_openai_api_version,
    azure_endpoint=config.azure_openai_api_endpoint,
    deployment_name="gpt-4o",
    max_retries=3
)

# embeddings model settings, use whichever
embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name="text-embedding-ada-002",
    api_key=config.azure_openai_api_embedding_key,
    azure_endpoint=config.azure_openai_api_embedding_endpoint,
    api_version=config.azure_openai_api_embedding_version
)
embed_model = AzureOpenAIEmbedding(
    model="text-embedding-3-large",         # use a larger model for better accuracy
    deployment_name="text-embedding-3-large",
    api_key=config.azure_openai_api_embedding_key,
    azure_endpoint=config.azure_openai_api_embedding_endpoint,
    api_version=config.azure_openai_api_embedding_version
)
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"     # free, lightweight, high-performing model
    # model_name="BAAI/bge-small-en"                        # free, open-source
    # model_name="sentence-transformers/all-mpnet-base-v2"  # free, better accuracy, larger, slower
)

# apply settings
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512       # chunks of 512 tokens
Settings.chunk_overlap = 50     # Overlap to retain context


class DocumentReader:
    def __init__(self, filename: str, file: bytes):
        self.filename = filename
        self.file = file

    async def _get_documents(self):
        pdf_reader = PDFText()
        pdf_reader.get_pdf_text(self.file)

        documents = []
        for page_num, content in pdf_reader.items():
            documents.append(
                Document(text=content, metadata={"filename": self.filename, "page_num": page_num})
            )
        
        return documents
    
    async def _get_query_engine(self, documents):
        pipeline = IngestionPipeline(
            transformations=[SentenceSplitter(chunk_size=512, chunk_overlap=50), embed_model]
        )
        nodes = await pipeline.arun(documents=documents, num_workers=3)
        index = VectorStoreIndex(nodes=nodes)

        storage_context = StorageContext.from_defaults()
        index = VectorStoreIndex.from_documents(
            documents=documents,
            storage_context=storage_context
        )

        # TODO: test and use either of the two indexes.
        
        # define reranker to prioritize most relevant chunks. reduces noise and improves accuracy
        reranker = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-12-v2", top_n=5)
        query_engine = index.as_query_engine(
            similarity_top_k=15,                    # initial retrieval
            node_postprocessors=[reranker],         # rerank to top 5
            response_mode="tree_summarize"
        )
        
        return query_engine
    
    async def query_llm(self, query: str):
        # create page-wise documents
        documents = await self._get_documents()

        # create query-engine over documents
        query_engine = await self._get_query_engine(documents=documents)

        # query over documents
        response = await query_engine.aquery(query)
        results = {
            "answer": response.response,
            "sources": [
                {"filename": node.metadata["filename"], "page_num": node.metadata["page_num"]}
                for node in response.source_nodes
            ]
        }

        return results
