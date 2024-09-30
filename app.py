import os
import logging
import threading
import pathway as pw
import litellm
import google.generativeai as genai
from pyngrok import ngrok
from pathlib import Path
from pathway.udfs import DiskCache, ExponentialBackoffRetryStrategy
from pathway.xpacks.llm.question_answering import BaseRAGQuestionAnswerer, RAGClient
from pathway.xpacks.llm.vector_store import VectorStoreServer
from pathway.xpacks.llm.embedders import GeminiEmbedder
from pathway.xpacks.llm import prompts, embedders, llms, parsers
from pathway.xpacks.llm.parsers import OpenParse
from pathway.xpacks.llm.llms import LiteLLMChat

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up API keys and environment variables
os.environ['LITELLM_LOG'] = 'DEBUG'
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-gemini-api-key-here")  # Use environment variables for secrets
os.environ['GEMINI_API_KEY'] = GEMINI_API_KEY
os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract/tessdata/"

# Configure Google Gemini AI
genai.configure(api_key=GEMINI_API_KEY)

# Set Pathway license key (demo for now)
pw.set_license_key("demo-license-key-with-telemetry")

# Document storage using Google Drive (replace with your own credentials if needed)
def load_gdrive_folder():
    folder = pw.io.gdrive.read(
        object_id="1kqq_GpBctFWak0bhUm-PxigGSNxDoBHm",
        service_user_credentials_file="research-papers-436618-d8072bb152e4.json",
        with_metadata=True
    )
    return [folder]

# Initialize LLM chat and embedder
chat = llms.LiteLLMChat(
    model="gemini/gemini-1.5-flash",
    retry_strategy=ExponentialBackoffRetryStrategy(max_retries=6, backoff_factor=2.5),
    temperature=0.0
)

embedder = GeminiEmbedder(
    model="models/embedding-001",
    retry_strategy=ExponentialBackoffRetryStrategy(max_retries=6, backoff_factor=2.5)
)

# Set up parsing arguments for tables and images
table_args = {
    "parsing_algorithm": "llm",
    "llm": chat,
    "prompt": prompts.DEFAULT_MD_TABLE_PARSE_PROMPT,
}

image_args = {
    "parsing_algorithm": "llm",
    "llm": chat,
    "prompt": prompts.DEFAULT_IMAGE_PARSE_PROMPT,
}

# Initialize parser
parser = parsers.OpenParse(table_args=table_args, image_args=image_args, parse_images=True)

# Document store setup
def initialize_document_store():
    sources = load_gdrive_folder()
    doc_store = VectorStoreServer(
        *sources,
        embedder=embedder,
        splitter=None,
        parser=parser,
    )
    return doc_store

# Initialize question answerer application
def initialize_app(doc_store):
    app = BaseRAGQuestionAnswerer(
        llm=chat,
        indexer=doc_store,
        search_topk=2,
        short_prompt_template=prompts.prompt_qa
    )
    return app

# Run server in a thread
def run_server(app):
    app_host = "0.0.0.0"
    app_port = 8000
    app.build_server(host=app_host, port=app_port)

    t = threading.Thread(target=app.run_server, name="BaseRAGQuestionAnswerer")
    t.daemon = True
    t.start()
    return app_host, app_port

# Expose server using ngrok
def expose_via_ngrok(port):
    public_url = ngrok.connect(port)
    logging.info(f"Public URL: {public_url}")
    return public_url

# Main function to run the application
def main():
    # Initialize document store
    doc_store = initialize_document_store()

    # Initialize and run the app
    app = initialize_app(doc_store)
    app_host, app_port = run_server(app)

    # Expose the app using ngrok
    public_url = expose_via_ngrok(app_port)

    # Initialize RAG client and test a query
    client = RAGClient(host=public_url, port=app_port)
    
    # Test with a sample question
    response = client.pw_ai_answer("What is calculus?")
    print(response)

if __name__ == "__main__":
    main()
