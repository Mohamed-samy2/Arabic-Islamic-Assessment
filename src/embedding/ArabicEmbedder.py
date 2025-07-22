import numpy as np
import torch
from abc import ABC, abstractmethod
from langchain_huggingface import HuggingFaceEmbeddings

class Embedder(ABC):
    def __init__(self, model_name, batch_size):
        self.model_name = model_name
        self.batch_size = batch_size

        self.device = ('cuda' if torch.cuda.is_available () else 'cpu')
        
        print("Using Devive", self.device)

        self.embedding_model = HuggingFaceEmbeddings(model_name = model_name, model_kwargs = {'device':self.device}, encode_kwargs = {'normalize_embeddings': True}, multi_process = False, show_progress= False, cache_folder= './embedder_model_cache')

    @abstractmethod
    def embed_documents(self, documents):
        pass
    @abstractmethod
    def embed_batch(self, text, batch_size = None):
        pass

class ArabicEmbedder(Embedder):
    def __init__(self, model_name, batch_size = 50):
        super().__init__(model_name, batch_size)

    def embed_documents(self, documents):
        return self.embed_batch(documents, batch_size = self.batch_size)

    def embed_batch(self, text, batch_size = None):
        if batch_size is None:
            batch_size = self.batch_size
            
        embeddings = []    
        for i in range(0, len(text), batch_size):
            batch = text[i:i + batch_size]
            batch_embeddings = self.embedding_model.embed_documents(batch)
            embeddings.extend(batch_embeddings)
        return embeddings


#pip install -U langchain-community
'''
embedder = ArabicEmbedder("CAMeL-Lab/bert-base-arabic-camelbert-mix", batch_size=16)

texts = ["ما هو الإسلام؟", "من هو النبي محمد؟", "ما هي أركان الإسلام؟"]
vectors = embedder.embed_batch(texts)

print(vectors.shape)
'''