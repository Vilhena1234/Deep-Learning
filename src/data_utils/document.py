"""
A generic document class 
"""
class Document:
    def __init__(self, text: str, _id: str) -> None:
        self.text = text
        self._id = _id
        
    def GetId(self):
        return self._id
    
    def GetText(self):
        return self.text