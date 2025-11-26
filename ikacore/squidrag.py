from abc import abstractmethod


class SquidRAGSource:
    def __init__(self, name:str, description:str, type:str):
        self.name = name
        self.description = description
        self.type = type

    @abstractmethod
    def search(self, query:str) -> list[str]:
        pass

    @abstractmethod
    def get_document(self, document_id:str) -> str:
        pass

    @abstractmethod
    def get_document_by_url(self, url:str) -> str:
        pass

    @abstractmethod
    def get_document_by_file(self, file:str) -> str:
        pass

    @abstractmethod
    def get_document_by_text(self, text:str) -> str:
        pass