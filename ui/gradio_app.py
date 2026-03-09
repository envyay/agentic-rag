from project.core.rag_system import RAGSystem


def create_gradio_ui():
    rag_system = RAGSystem()
    rag_system.initialize()

    # doc_manager = DocumentManager(rag_system)
    # chat_interface = ChatInterface(rag_system)