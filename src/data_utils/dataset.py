from src.data_utils.document import Document

class DocumentDataset:
    def __init__(self, documents: list[dict]) -> None:
        self.documents: list[Document] = self.__BuildDocuments(documents)
        
    def __BuildDocuments(self, documents):
        
        return [Document(text=text, _id=_id) for entries in documents 
                    for (text, _id) in zip(entries['text'], entries['document_id'])
                ]
        
    def GetDocuments(self) -> list[Document]:
        return self.documents