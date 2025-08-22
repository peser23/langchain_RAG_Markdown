from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from dotenv import load_dotenv
import os

DATA_PATH = "data/books"
CHROMA_PATH = "db/chroma"

load_dotenv()

def load_documents():
    """Load documents from the specified directory."""
    loader = DirectoryLoader(DATA_PATH, glob="**/*.md")
    documents = loader.load()
    return documents

def split_text(documents: list[Document]):
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500, length_function=len, add_start_index=True)
    chunks = text_splitter.split_documents(documents)
    print(f"total documents: {len(documents)}")
    print(f"Number of chunks created: {len(chunks)}")
    return chunks

def create_vector_store(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        print(f"Removing existing directory: {CHROMA_PATH}")
        os.rmdir(CHROMA_PATH)

    # Create embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Create a Chroma vector store
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)

    # Save the vector store to disk
    vector_store.persist()
    print(f"Saved {len(chunks)} Vector store saved to {CHROMA_PATH}")

def main():
    # Load documents
    documents = load_documents()
    chunks = split_text(documents)
    create_vector_store(chunks)

if __name__ == "__main__":
    main()
 






