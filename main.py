import os
import argparse
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

class MarkdownSearcher:
    def __init__(self, persist_dir="chroma_db", use_openai=True):
        """Initialize the ChromaDB client with the correct settings.
        
        Args:
            persist_dir (str): Directory where ChromaDB files are stored
            use_openai (bool): Whether to use OpenAI embeddings (requires API key)
        """
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_dir)
        
        # Set up embedding function
        if use_openai:
            # Load environment variables
            load_dotenv()
            
            # Get OpenAI API key
            openai_api_key = os.getenv('OPENAI_API_KEY')
            if not openai_api_key:
                print("Warning: OpenAI API key not found. Falling back to default embeddings.")
                self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction()
                print("Using default SentenceTransformer embeddings.")
            else:
                # Create OpenAI embedding function
                self.embedding_func = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=openai_api_key,
                    model_name="text-embedding-3-small"
                )
                print("Using OpenAI text-embedding-3-small model for embeddings.")
        else:
            # Use default sentence transformer embeddings
            self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction()
            print("Using default SentenceTransformer embeddings.")
        
        # Get collection
        try:
            self.collection = self.client.get_collection(
                name="markdown_docs",
                embedding_function=self.embedding_func
            )
            print(f"Connected to existing collection 'markdown_docs'")
        except Exception as e:
            print(f"Error: Could not connect to collection: {str(e)}")
            print("Make sure you've indexed documents before searching.")
            raise

    def search(self, query, n_results=5, path_filter=None, tag_filter=None):
        """Search for documents matching the query.
        
        Args:
            query (str): The search query
            n_results (int): Number of results to return
            path_filter (str, optional): Filter results by file path
            tag_filter (str, optional): Filter results by tag
            
        Returns:
            dict: Search results with documents, metadata, and distances
        """
        # Perform the search
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Filter results if needed
        if path_filter or tag_filter:
            filtered_docs = []
            filtered_metadatas = []
            filtered_distances = []
            
            for i, metadata in enumerate(results['metadatas'][0]):
                # Check path filter
                if path_filter and path_filter not in metadata.get('source_path', ''):
                    continue
                
                # Check tag filter
                if tag_filter:
                    tags = metadata.get('tags', [])
                    if isinstance(tags, str):
                        tags = [t.strip() for t in tags.split(',')]
                    if tag_filter not in tags:
                        continue
                
                # Include this result
                filtered_docs.append(results['documents'][0][i])
                filtered_metadatas.append(metadata)
                filtered_distances.append(results['distances'][0][i])
            
            # Replace results with filtered results
            filtered_results = {
                'documents': [filtered_docs],
                'metadatas': [filtered_metadatas],
                'distances': [filtered_distances],
                'ids': results['ids']  # Keep original IDs
            }
            return filtered_results
        
        return results

    def display_results(self, results, query):
        """Display search results in a readable format.
        
        Args:
            results (dict): Search results from the search method
            query (str): The original search query
        """
        if not results['documents'][0]:
            print("\nNo results found matching your criteria.")
            return
            
        print(f"\n{'=' * 80}")
        print(f"Found {len(results['documents'][0])} results for: '{query}'")
        print(f"{'=' * 80}")
        
        for i, (doc, metadata, dist) in enumerate(zip(
            results['documents'][0], 
            results['metadatas'][0],
            results['distances'][0]
        )):
            # Calculate similarity score (1 - distance for cosine distance)
            similarity = round((1 - dist) * 100, 2)
            
            # Get source file info
            source = metadata.get('source_path', 'Unknown')
            
            # Print result with formatting
            print(f"\n----- Result {i+1} (Similarity: {similarity}%) -----")
            print(f"Source: {source}")
            
            # Display metadata fields
            if 'title' in metadata:
                print(f"Title: {metadata['title']}")
            if 'tags' in metadata:
                if isinstance(metadata['tags'], list):
                    print(f"Tags: {', '.join(metadata['tags'])}")
                else:
                    print(f"Tags: {metadata['tags']}")
            if 'date' in metadata:
                print(f"Date: {metadata['date']}")
                
            # Print chunk information
            if 'chunk_index' in metadata and 'total_chunks' in metadata:
                print(f"Chunk: {metadata['chunk_index'] + 1}/{metadata['total_chunks']}")
            
            # Content preview
            preview = doc[:300] + "..." if len(doc) > 300 else doc
            print(f"\n{preview}")
            print(f"\n{'-' * 80}")

def interactive_search(searcher, path_filter=None, tag_filter=None, n_results=5):
    """Run an interactive search session.
    
    Args:
        searcher (MarkdownSearcher): The searcher instance
        path_filter (str, optional): Filter results by file path
        tag_filter (str, optional): Filter results by tag
        n_results (int): Number of results to return
    """
    print("\n=== Markdown Search ===")
    if path_filter:
        print(f"Path filter: {path_filter}")
    if tag_filter:
        print(f"Tag filter: {tag_filter}")
    print("Type 'exit' to quit\n")
    
    while True:
        query = input("Enter your query: ")
        if query.lower() == 'exit':
            break
        
        try:
            print("\nSearching...")
            results = searcher.search(
                query=query, 
                n_results=n_results,
                path_filter=path_filter,
                tag_filter=tag_filter
            )
            searcher.display_results(results, query)
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nExiting search. Goodbye!")

def main():
    parser = argparse.ArgumentParser(description='Search indexed markdown documents')
    parser.add_argument('--db-path', default='chroma_db', help='ChromaDB persistence directory')
    parser.add_argument('--use-openai', action='store_true', help='Use OpenAI embeddings (requires API key)')
    parser.add_argument('--n-results', type=int, default=5, help='Number of results to return')
    parser.add_argument('--path-filter', help='Filter results by file path')
    parser.add_argument('--tag-filter', help='Filter results by tag')
    parser.add_argument('--query', help='Search query (omit for interactive mode)')
    
    args = parser.parse_args()
    
    try:
        # Initialize searcher
        searcher = MarkdownSearcher(
            persist_dir=args.db_path,
            use_openai=args.use_openai
        )
        
        # Either do a single query or run in interactive mode
        if args.query:
            # Single query mode
            results = searcher.search(
                query=args.query,
                n_results=args.n_results,
                path_filter=args.path_filter,
                tag_filter=args.tag_filter
            )
            searcher.display_results(results, args.query)
        else:
            # Interactive mode
            interactive_search(
                searcher=searcher,
                path_filter=args.path_filter,
                tag_filter=args.tag_filter,
                n_results=args.n_results
            )
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()
