import uuid

from langchain_community.chat_models import ChatOllama

from project import config
from project.db.parent_store_manager import ParentStoreManager
from project.db.vector_db_manager import VectorDbManager
from project.document_chunker import DocumentChuncker


class RAGSystem:

    def __init__(self, collection_name=config.CHILD_COLLECTION):
        self.collection_name = collection_name
        self.vector_db = VectorDbManager()
        self.parent_store = ParentStoreManager()
        self.chunker = DocumentChuncker()
        self.agent_graph = None
        self.thread_id = str(uuid.uuid4())
        self.recursion_limit = 50

    def initialize(self):
        self.vector_db.create_collection(self.collection_name)
        collection = self.vector_db.get_collection(self.collection_name)

        llm = ChatOllama(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)
        tools = ToolFactory(collection).create_tools()
        self.agent_graph = create_agent_graph(llm, tools)


