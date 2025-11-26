from tools import SquidTools
from squidrag import SquidRAGSource
class SquidStage:
    def __init__(self, name:str, description:str, tools:list[SquidTools], RAGSource:list[type[SquidRAGSource]]=[]):
        self.name = name
        self.description = description
        self.tools = tools
        self.RAGSource = RAGSource # list of RAG source types to use