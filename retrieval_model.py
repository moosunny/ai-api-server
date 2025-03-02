from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings # Select embeddings model
from langchain_core.vectorstores import InMemoryVectorStore
from langchain import hub
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

class Retrieval_Model:
    def __init__(self):
        self.model = init_chat_model("gpt-4o-mini", model_provider="openai")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.vector_store = InMemoryVectorStore(embeddings)

        loader = WebBaseLoader(
            web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            ),
        )

        docs = loader.load()

        # Text splitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(docs)

        # Index chunks
        _ = self.vector_store.add_documents(documents=all_splits)

        # Define prompt for question-answering
        self.prompt = hub.pull("rlm/rag-prompt")

        # Define application steps
        def retrieve(state: State):
            retrieved_docs = self.vector_store.similarity_search(state["question"])
            return {"context": retrieved_docs}

        def generate(state: State):
            docs_content = "\n\n".join(doc.page_content for doc in state["context"])
            messages = self.prompt.invoke({"question": state["question"], "context": docs_content})
            response = self.model.invoke(messages)
            return {"answer": response.content}
        
        # Compile application and test
        self.graph = StateGraph(State)
        self.graph.add_node("retrieve", retrieve)
        self.graph.add_node("generate", generate)
        self.graph.add_edge(START, "retrieve")
        self.graph.add_edge("retrieve", "generate")
        self.graph.set_entry_point("retrieve")
        self.graph = self.graph.compile()
    
    def get_response(self, question, thred_id):
        config = {"configurable": {"thread_id": thred_id}}
        response = self.graph.invoke({"question": question}, config)
        return response["answer"]