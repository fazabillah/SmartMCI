
from langchain.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

load_dotenv(override=True)


pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Ingestion for API 571,970, and 584 Documents
api_documents = {
    "api571": {
        "file_path": "data/raw_documents/api571_damage_mechanisms.pdf",
        "index_name": "api571-damage-mechanisms",
        "description": "API 571 - Damage Mechanisms Affecting Fixed Equipment"
    },
    "api970": {
        "file_path": "data/raw_documents/api970_corrosion_control.pdf", 
        "index_name": "api970-corrosion-control",
        "description": "API 970 - Corrosion Control Document"
    },
    "api584": {
        "file_path": "data/raw_documents/api584_integrity_windows.pdf", 
        "index_name": "api584-integrity-windows", 
        "description": "API 584 - Integrity Operating Windows"
    }
}

#Step 1: Setup embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

#Step 2: Process each API document
def process_api_document(api_key, config):
    """Process a single API document and upload to Pinecone"""
    
    print(f"\n🔄 Processing {config['description']}")
    
    # Check if file exists
    if not os.path.exists(config['file_path']):
        print(f"❌ File not found: {config['file_path']}")
        return False
    
    # Load PDF document
    loader = PyPDFLoader(config['file_path'])
    documents = loader.load()
    print(f"📄 Loaded {len(documents)} pages")
    
    # Chunking documents into smaller parts
    documents = text_splitter.split_documents(documents)
    print(f"✂️ Created {len(documents)} chunks")
    
    # Add metadata to each chunk
    for i, doc in enumerate(documents):
        doc.metadata.update({
            "api_standard": api_key.upper(),
            "document_type": get_document_type(api_key),
            "chunk_id": f"{api_key}_{i:04d}",
            "source_file": os.path.basename(config['file_path'])
        })
    
    # Create or get Pinecone index
    index_name = config['index_name']
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        print(f"🆕 Creating new index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=768,  # all-mpnet-base-v2 dimension
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws", 
                region="us-east-1"
            )
        )
    else:
        print(f"📊 Using existing index: {index_name}")
    
    # Get index and create vector store
    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
    )
    
    # Upload documents to Pinecone
    result = vector_store.add_documents(documents)
    print(f"✅ Added {len(result)} documents to Pinecone index '{index_name}'")
    
    return True

def get_document_type(api_key):
    """Get document type for metadata"""
    type_mapping = {
        "api571": "damage_mechanism_identification",
        "api970": "corrosion_control_strategy",
        "api584": "integrity_operating_windows"
    }
    return type_mapping.get(api_key, "unknown")


# Step 3: Process all API documents
print("🚀 Starting API Documents Processing")
print("=" * 50)

success_count = 0
total_count = len(api_documents)

for api_key, config in api_documents.items():
    try:
        success = process_api_document(api_key, config)
        if success:
            success_count += 1
    except Exception as e:
        print(f"❌ Error processing {api_key.upper()}: {e}")


# Step 4: Summary and verification
print("\n" + "=" * 50)
print("📊 PROCESSING SUMMARY")
print("=" * 50)

print(f"✅ Successfully processed: {success_count}/{total_count} documents")

if success_count == total_count:
    print("🎉 All API documents processed successfully!")
    print("📋 You can now run the main application")
else:
    print("⚠️ Some documents failed to process. Check the errors above.")

# Verify uploads
print("\n🔍 Verification:")
for api_key, config in api_documents.items():
    try:
        index_name = config['index_name']
        if index_name in [index_info["name"] for index_info in pc.list_indexes()]:
            index = pc.Index(index_name)
            stats = index.describe_index_stats()
            vector_count = stats.get('total_vector_count', 0)
            print(f"📊 {api_key.upper()}: {vector_count} vectors in '{index_name}'")
        else:
            print(f"❌ {api_key.upper()}: Index not found")
    except Exception as e:
        print(f"❌ {api_key.upper()}: Error checking index - {e}")

print(f"\n📝 Processing complete. Check logs above for any issues.")
print(f"💡 Note: Run this script only once or when updating documents.")


