from helpers.config import get_settings
from vectordb.Qdrantdb import Qdrantdb
from embedding.ArabicEmbedder import ArabicEmbedder
from text_processing.ArabicTextPreprocessor import ArabicTextPreprocessor
import os

def main():
    
    settings = get_settings()
    data_path = settings.Data_Path
    
    preprocessor = ArabicTextPreprocessor(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
    db = Qdrantdb(db_path=settings.Qdrant_db_path,distance_method=settings.Qdrant_distance_method)
    embedding_model = ArabicEmbedder(settings.Embedding_Model_Name,settings.BATCH_SIZE)
    db.connect()
    
    is_created = db.create_collection(collection_name=settings.DB_NAME,embedding_size=settings.Embedding_Model_Size,do_reset=settings.DO_RESET)
    
    if is_created:
        print("Collection Created")
        number_of_files = len(os.listdir(data_path))
        print(f"Number of files: {number_of_files}")
        
        i = 1 
    
        for file in os.listdir(data_path):
            file_path = os.path.join(data_path, file)

            chunks = preprocessor.process_document(
                file_path=file_path,
                chunking_method='sentence_based',
                cleaning_options={
                    'remove_diacritics': True,
                    'remove_tatweel': True,
                    'normalize_punctuation': True,
                    'remove_non_arabic': False,
                    'normalize_whitespace': True,
                    # Enable new flags here if desired:
                    'remove_phone_numbers': True,
                    'remove_emails': True,
                    'remove_flixat': True
                },
                use_uploaded_file=False
            )
            
            print(f"\n📊 Processing Results:")
            print(f"   • Total chunks: {len(chunks)}")
            
            cleaned_texts = [chunk.cleaned_text for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            
            embeddings = embedding_model.embed_documents(cleaned_texts)


            _ = db.insert_documents(collection_name=settings.DB_NAME,documents=cleaned_texts,embedding_vectors=embeddings,metadata=metadatas,batch_size=settings.BATCH_SIZE)
            
            if not _:
                print(f"Error inserting documents")
                break
            
            print(f"Finished {i}/{number_of_files}")
            i+=1

    query_vector =  embedding_model.embed_documents([query])
    results = db.search_by_vector(collection_name=settings.DB_NAME,query_vector=query_vector[0],top_k=20)

main()