from qdrant_client import models,QdrantClient
from qdrant_client.http.models import HnswConfig
from typing import List
import os

class Qdrandb:
    def __init__(self,db_path:str,distance_method:str="cosine"):
        
        self.client=None
        base_path = os.path.dirname(os.path.dirname(__file__))
        
        self.db_path = os.path.join(base_path,db_path)
        
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)
        
        if distance_method == "dot":
            self.distance_method = models.Distance.DOT
        elif distance_method == "cosine":
            self.distance_method = models.Distance.COSINE
        elif distance_method == "euclidean":
            self.distance_method = models.Distance.EUCLID
    
    
    
    def connect(self):
        self.client = QdrantClient(path=self.db_path)
    
    def disconnect(self):
        pass
    
    def create_collection(self,collection_name:str,embedding_size:int,do_reset:bool=False):
        
        if do_reset:
            _= self.delete_collection(collection_name)
            print(f"Collection {collection_name} deleted")

        if self.is_collection_exists(collection_name):
            print(f"Collection {collection_name} already exists")
            return False
        
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=embedding_size,
                                            distance=self.distance_method,
                                            hnsw_config=HnswConfig(
                                                m=32,
                                                ef_construct=128,
                                                full_scan_threshold=512,
                                                payload_m=32)
                                            ),
            timeout=20,
        )
        print(f"Collection {collection_name} created")
        
        return True
    
    def is_collection_exists(self,collection_name:str):
        return self.client.collection_exists(collection_name=collection_name)
    
    def delete_collection(self,collection_name:str):
        
        if self.is_collection_exists(collection_name):
            return self.client.delete_collection(collection_name=collection_name)
    
    def insert_documents(self, collection_name:str, documents:List[str],
                        embedding_vectors:List[List[float]], metadata:List[dict],batch_size: int= 100):
        
        if metadata is None:
            metadata = [{}] * len(documents)
            
        
        for i in range(0, len(documents), batch_size):
            batch_documents = documents[i:i + batch_size]
            batch_embedding_vectors = embedding_vectors[i:i + batch_size]
            batch_metadata = metadata[i:i + batch_size]
            
            batch_records =[
                
                models.Record(
                    vector=batch_embedding_vectors[x],
                    payload={
                        "text": batch_documents[x],
                        "metadata": batch_metadata[x],
                        }
                )
                for x in range(len(batch_documents))
            ]
            try:
                _= self.client.upload_records(
                    collection_name=collection_name,
                    batch_size=batch_size,
                    records=batch_records,
                    parallel=3,
                    wait=True,
                )
            except Exception as e:
                print(e)
                return False
            
        print(f"Inserted {len(batch_documents)} records into collection {collection_name}")
        return True
    
    def search_by_vector(self,collection_name:str,vector:List[float], top_k:int=20, filter:dict=None):
        
        return self.client.search(
            collection_name=collection_name,
            query_vector=vector,
            limit=top_k,
            filter=filter,
            with_payload=True,
            with_vectors=False,
        )
        