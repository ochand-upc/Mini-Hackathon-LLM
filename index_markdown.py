import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from markdown import markdown
import frontmatter
from tqdm import tqdm
from typing import List, Dict, Tuple
import argparse
from bs4 import BeautifulSoup
import re
from dotenv import load_dotenv
from openai import OpenAI

class MarkdownIndexer:
    def __init__(self, persist_dir: str = "chroma_db", chunk_size: int = 500, chunk_overlap: int = 50, use_openai: bool = False):
        """Initialize the ChromaDB client with persistence.
        
        Args:
            persist_dir (str): Directory to store ChromaDB files
            chunk_size (int): Maximum number of characters per chunk
            chunk_overlap (int): Number of characters to overlap between chunks
            use_openai (bool): Whether to use OpenAI embeddings (requires API key)
        """
        # Set up embedding function
        if use_openai:
            # Load environment variables
            load_dotenv()
            
            # Initialize OpenAI client
            openai_api_key = os.getenv('OPENAI_API_KEY')
            if not openai_api_key:
                print("Warning: OpenAI API key not found. Falling back to default embeddings.")
                embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction()
                print("Using default SentenceTransformer embeddings.")
            else:
                # Create OpenAI embedding function
                embedding_func = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=openai_api_key,
                    model_name="text-embedding-3-small"
                )
                print("Using OpenAI text-embedding-3-small model for embeddings.")
        else:
            # Use default sentence transformer embeddings
            embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction()
            print("Using default SentenceTransformer embeddings.")
        self.client = chromadb.PersistentClient(path=persist_dir)
        
        # Try to get existing collection
        collection_name = "markdown_docs"
        try:
            existing_collection = self.client.get_collection(name=collection_name)
            # If collection exists, try adding a document to check dimensions
            try:
                existing_collection.add(
                    documents=["test"],
                    metadatas=[{"test": "test"}],
                    ids=["test"],
                )
                # If successful, delete test document
                existing_collection.delete(ids=["test"])
                self.collection = existing_collection
                print(f"Using existing collection '{collection_name}'")
            except Exception as e:
                # If dimensions don't match, delete and recreate collection
                if "dimension" in str(e).lower():
                    print(f"Recreating collection '{collection_name}' due to embedding dimension change")
                    self.client.delete_collection(name=collection_name)
                    self.collection = self.client.create_collection(
                        name=collection_name,
                        metadata={"hnsw:space": "cosine"},
                        embedding_function=embedding_func
                    )
                else:
                    raise e
        except Exception as e:
            # Collection doesn't exist, create it
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=embedding_func
            )
            print(f"Created new collection '{collection_name}'")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def create_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks.
        
        Args:
            text (str): Text to split into chunks
            
        Returns:
            List[str]: List of text chunks
        """
        # Split text into sentences (basic splitting)
        sentences = re.split(r'(?<=[.!?]) +', text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Join the current chunk and add it to chunks
                chunks.append(' '.join(current_chunk))
                # Keep last sentence if overlap is needed
                overlap_sentences = []
                overlap_size = 0
                for s in reversed(current_chunk):
                    if overlap_size + len(s) > self.chunk_overlap:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_size += len(s)
                current_chunk = overlap_sentences
                current_size = overlap_size
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def process_markdown_file(self, file_path: str) -> List[Dict]:
        """Process a markdown file and extract its content and metadata.
        
        Returns a list of chunks with their metadata.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            post = frontmatter.load(f)
        
        # Extract frontmatter metadata
        metadata = dict(post.metadata)
        
        # Convert markdown to HTML first, then use BeautifulSoup to get clean text
        html_content = markdown(post.content)
        soup = BeautifulSoup(html_content, 'html.parser')
        content = soup.get_text(separator=' ', strip=True)
        
        # Add file path to metadata
        metadata['source_path'] = file_path
        
        # Create chunks
        chunks = self.create_chunks(content)
        
        # Create a list of chunks with metadata
        chunk_documents = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata['chunk_index'] = i
            chunk_metadata['total_chunks'] = len(chunks)
            chunk_documents.append({
                'content': chunk,
                'metadata': chunk_metadata
            })
        
        return chunk_documents

    def index_directory(self, directory_path: str):
        """Index all markdown files in the given directory and its subdirectories."""
        markdown_files = []
        
        # Collect all markdown files
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith(('.md', '.markdown')):
                    markdown_files.append(os.path.join(root, file))
        
        if not markdown_files:
            print("No markdown files found in the specified directory.")
            return

        # Process each markdown file
        documents = []
        metadatas = []
        ids = []

        for file_path in tqdm(markdown_files, desc="Processing markdown files"):
            try:
                processed_chunks = self.process_markdown_file(file_path)
                
                # Generate base ID from file path
                base_id = os.path.relpath(file_path, directory_path).replace('/', '_')
                
                # Add each chunk with its metadata
                for chunk_doc in processed_chunks:
                    chunk_index = chunk_doc['metadata']['chunk_index']
                    # Create unique ID for each chunk
                    chunk_id = f"{base_id}_chunk_{chunk_index}"
                    
                    documents.append(chunk_doc['content'])
                    metadatas.append(chunk_doc['metadata'])
                    ids.append(chunk_id)
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")

        # Add documents to ChromaDB
        if documents:
            print("\nAdding documents to ChromaDB...")
            # Add in batches to avoid memory issues
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                end_idx = min(i + batch_size, len(documents))
                self.collection.add(
                    documents=documents[i:end_idx],
                    metadatas=metadatas[i:end_idx],
                    ids=ids[i:end_idx]
                )
            print(f"Successfully indexed {len(documents)} chunks from {len(markdown_files)} files")

    def query_documents(self, query_text: str, n_results: int = 5) -> List[Dict]:
        """Query the indexed documents."""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results

def main():
    parser = argparse.ArgumentParser(description='Index and query markdown files using ChromaDB')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Index markdown files')
    index_parser.add_argument('directory', help='Directory containing markdown files')
    index_parser.add_argument('--db-path', default='chroma_db', help='ChromaDB persistence directory')
    index_parser.add_argument('--chunk-size', type=int, default=500, help='Maximum number of characters per chunk')
    index_parser.add_argument('--chunk-overlap', type=int, default=50, help='Number of characters to overlap between chunks')
    index_parser.add_argument('--use-openai', action='store_true', help='Use OpenAI embeddings (requires API key)')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Search indexed documents')
    query_parser.add_argument('query', help='Search query')
    query_parser.add_argument('--db-path', default='chroma_db', help='ChromaDB persistence directory')
    query_parser.add_argument('--n-results', type=int, default=5, help='Number of results to return')
    query_parser.add_argument('--use-openai', action='store_true', help='Use OpenAI embeddings (requires API key)')
    query_parser.add_argument('--path-filter', help='Filter results by file path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    indexer = MarkdownIndexer(
        persist_dir=args.db_path,
        chunk_size=getattr(args, 'chunk_size', 500),
        chunk_overlap=getattr(args, 'chunk_overlap', 50),
        use_openai=args.use_openai
    )
    
    if args.command == 'index':
        indexer.index_directory(args.directory)
    elif args.command == 'query':
        results = indexer.query_documents(
            query_text=args.query,
            n_results=args.n_results
        )
        
        # Print results
        print(f"\nFound {len(results['documents'][0])} results:")
        for i, (document, metadata, score) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            # Filter by path if specified
            if args.path_filter and args.path_filter not in metadata['source_path']:
                continue
                
            print(f"\nResult {i+1} (similarity: {1-score:.3f})")
            print(f"Source: {metadata['source_path']}")
            print(f"Chunk: {metadata['chunk_index']+1}/{metadata['total_chunks']}")
            print("Content:", document)

if __name__ == "__main__":
    main()
