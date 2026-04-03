import os
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

VECTOR_STORE_PATH = "faiss_index"

def build_or_load_vector_store():
    # 🚀 We use a local Hugging Face model! Free, secure, and no rate limits.
    print("⏳ Initializing local Hugging Face embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # If the index already exists, load it directly
    if os.path.exists(VECTOR_STORE_PATH):
        print("✅ Loading existing FAISS Medical Knowledge Base")
        return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        
    print("⏳ Building new FAISS Medical Knowledge Base from CSVs (This might take 10-20 seconds)...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    desc_path = os.path.join(current_dir, "../data/symptom_Description.csv")
    prec_path = os.path.join(current_dir, "../data/symptom_precaution.csv")
    
    docs = []
    
    # Process Disease Descriptions
    if os.path.exists(desc_path):
        df_desc = pd.read_csv(desc_path)
        for _, row in df_desc.iterrows():
            disease = str(row.iloc[0]).strip()
            desc = str(row.iloc[1]).strip()
            content = f"Disease: {disease}\nDescription: {desc}"
            docs.append(Document(page_content=content, metadata={"disease": disease, "type": "description"}))
            
    # Process Disease Precautions (Recommendations)
    if os.path.exists(prec_path):
        df_prec = pd.read_csv(prec_path)
        for _, row in df_prec.iterrows():
            disease = str(row.iloc[0]).strip()
            precs = [str(x).strip().capitalize() for x in row.iloc[1:] if pd.notna(x)]
            content = f"Disease: {disease}\nPrecautions: {', '.join(precs)}"
            docs.append(Document(page_content=content, metadata={"disease": disease, "type": "precaution"}))
    
    if not docs:
        print("⚠️ Warning: No medical CSV data found to build vector store.")
        return None
        
    # Build and save the FAISS vector store
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)
    print("✅ FAISS Medical Knowledge Base built and saved successfully")
    return vector_store

# Initialize it once when the module loads
vector_store = build_or_load_vector_store()

def get_medical_context(disease_name):
    """Queries the Vector DB for descriptions and precautions related to a disease."""
    if not vector_store:
        return {"description": "Medical database unavailable.", "precautions": []}
        
    # Retrieve top 4 most relevant documents via semantic search
    results = vector_store.similarity_search(f"Disease: {disease_name}", k=4)
    
    description = ""
    precautions = []
    
    for doc in results:
        # Check metadata to ensure we matched the correct disease
        if doc.metadata.get("disease", "").lower() == disease_name.lower():
            if doc.metadata.get("type") == "description":
                description = doc.page_content.split("Description: ")[-1]
            elif doc.metadata.get("type") == "precaution":
                prec_text = doc.page_content.split("Precautions: ")[-1]
                precautions = [p.strip() for p in prec_text.split(",")]
    
    return {
        "description": description if description else "No detailed description available.",
        "precautions": precautions
    }