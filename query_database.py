from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "db/chroma"

def combine_documents(documents: list[Document]):
    """Combine documents into a single string."""
    combined_text = "\n\n------------\n\n".join(doc.page_content for doc, _score in documents)
    sources = [f"{doc.metadata.get("source", "Unknown source")}||{_score}"  for doc, _score in documents]
    return combined_text, sources

def query_vector_store(query: str, vector_store_path: str):
    """Query the vector store for similar documents."""
    enbeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # Load the vector store
    vector_store = Chroma(persist_directory=vector_store_path, embedding_function=enbeddings)

    # Perform the query
    results = vector_store.similarity_search_with_relevance_scores(query, k=3)  # Adjust k as needed
    if len(results) == 0 or results[0][1] < 0.1:
        print("Unable to find relevant results.")
        return

    return results


def generate_prompt(query: str, context: str):
    """Generate a prompt for the LLM based on the query."""
    prompt = PromptTemplate(        
        template="Answer the question based only on the following context:{context}\n\n" \
        "Question: {query}",
        input_variables=["query","context"],
    )
    return prompt.invoke({"query": query, "context": context})

def query_llm(query: str, context: str):
    """Query the LLM with the generated prompt."""
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
    prompt = generate_prompt(query, context)
    response = llm.invoke(prompt)
    return response

def main():
    query = input("Enter your query: ")

    results = query_vector_store(query, CHROMA_PATH)
    if not results:
        print("No relevant documents found.")
        return
    print(f"Found {len(results)} results for query: '{query}'")

    combined_text, sources = combine_documents(results)
    #print(f"Combined text from results:\n{combined_text}\n")

    results = query_llm(query, combined_text)
    print(f"Query: {query}\n")
    print(f"LLM response: {results.content}\n\nSources: {sources}")

if __name__ == "__main__":
    main()