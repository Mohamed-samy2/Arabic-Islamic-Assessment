from helpers.config import get_settings
from vectordb.Qdrantdb import Qdrandb


def main():
    
    settings = get_settings()
    
    db = Qdrandb(db_path=settings.Qdrant_db_path,distance_method=settings.Qdrant_distance_method)
    
    db.connect()
    
    db.create_collection(collection_name=settings.DB_NAME,embedding_size=settings.Embedding_Model_Size,do_reset=settings.DO_RESET)
    
    db.insert_documents(collection_name=settings.DB_NAME,documents=["Hello World"],embedding_vectors=[[1]*settings.Embedding_Model_Size],metadata=[{"text":"Hello World"}])


main()