"""
Qdrant Vector Database Utilities

Generic, reusable Qdrant vector database functions that can be used by any pipeline.
Supports three modes: local (disk), memory (in-memory), and docker (containerized).
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, HnswConfigDiff

# -------------------------------------------------------------------
@dataclass
class QdrantConfig:
    """
    Configuration for Qdrant vector database
    
    Supports both single-vector and multi-vector modes.
    
    Example single-vector:
        config = QdrantConfig(
            embedding_dimension=384,
            distance="COSINE"
        )
    
    Example multi-vector:
        config = QdrantConfig(
            vectors_config={
                "text": {"dimension": 384, "distance": "COSINE"},
                "image": {"dimension": 512, "distance": "DOT"}
            }
        )
    """
    
    # Database mode: 'docker', 'memory', or 'local'
    db_mode: str = "local"
    
    # Local disk path (used when db_mode = "local")
    local_path: str = "./local_qdrant"
    
    # Docker connection settings
    docker_url: str = "http://localhost:6333"
    
    # Collection settings
    collection_name: str = "documents"
    
    # Single-vector mode (used if vectors_config is None)
    embedding_dimension: int = 384
    distance: str = "COSINE"  # COSINE, DOT, MANHATTAN, EUCLID
    
    # Multi-vector mode (overrides single-vector if specified)
    # Format: {"vector_name": {"dimension": int, "distance": str}}
    # Example: {"text": {"dimension": 384, "distance": "COSINE"}}
    vectors_config: Optional[Dict[str, Dict]] = None
    
    # HNSW Index Parameters
    # Example:
    # {"m": 16, "ef_construct": 200, "full_scan_threshold": 10000, "on_disk": False}
    hnsw_config: Optional[Dict] = None
    
    # Search-time parameters
    # Example: {"ef": 100}
    search_params: Optional[Dict] = None
    
    # Retrieval settings
    top_k: int = 5

# -------------------------------------------------------------------
class QdrantVectorDB:
    """
    Generic Qdrant Vector Database for storage and retrieval.
    
    This class handles:
    - Initialization of Qdrant client based on configuration (local, memory, docker)
    - Creation of collections with single or multiple vector fields
    - Indexing of documents with pre-computed embeddings
    - Retrieval of similar documents using pre-computed query embeddings
        - Utility functions for collection management (count, info, clear)
    
    """
    #-------------------
    def __init__(self, config: QdrantConfig):
        """
        Initialize the vector database.
        
        Args:
            config: QdrantConfig object with database settings
        """
        self.config = config # Store the configuration for later use in collection creation and retrieval
        self.client = self._initialize_client() # Initialize the Qdrant client based on the specified mode (local, memory, docker)
    #-------------------
    def _get_distance_metric(self, distance_str: str) -> Distance:
        """Convert distance string to Distance enum."""
        distance_map = {
            "COSINE": Distance.COSINE,
            "DOT": Distance.DOT,
            "MANHATTAN": Distance.MANHATTAN,
            "EUCLID": Distance.EUCLID,
        }
        return distance_map.get(distance_str.upper(), Distance.COSINE)
    #-------------------
    def _build_vectors_config(self) -> Dict:
        """Build vectors configuration for collection creation."""
        
        # If vectors_config is provided, we are in multi-vector mode and need to build a dict of vector_name -> VectorParams
        if self.config.vectors_config:
            # Multi-vector mode
            vectors_config = {}
            
            # Loop through each vector field in the config and create a VectorParams object for it
            for vector_name, params in self.config.vectors_config.items():
                dimension = params.get("dimension", 384)           # Default dimension if not specified
                distance_str = params.get("distance", "COSINE")    # Default distance if not specified
                distance = self._get_distance_metric(distance_str) # Convert distance string to Distance enum
                # For multi-vector collections, we use a dict of vector_name -> VectorParams
                vectors_config[vector_name] = VectorParams(
                    size=dimension,
                    distance=distance
                )
            return vectors_config
        else:
            # Single-vector mode
            distance = self._get_distance_metric(self.config.distance) # Convert distance string to Distance enum
            return VectorParams( # For single-vector collections, we return a single VectorParams object
                size=self.config.embedding_dimension,
                distance=distance
            )
    #-------------------
    def _initialize_client(self) -> QdrantClient:
        """Initialize Qdrant client based on config mode."""
        mode = self.config.db_mode.lower() # Get the database mode from the configuration and convert it to lowercase for consistency
        
        if mode == "docker":
            print(f"Connecting to Qdrant Docker at {self.config.docker_url}...")
            client = QdrantClient(url=self.config.docker_url) # Connect to the Qdrant instance running in Docker using the specified URL
            print("✓ Connected to Qdrant Docker")

        elif mode == "memory":
            print("Initializing in-memory Qdrant...")
            client = QdrantClient(":memory:") # Initialize an in-memory Qdrant instance (data will not persist after the program ends)
            print("[OK] In-memory Qdrant initialized")

        elif mode == "local":
            print(f"Initializing local Qdrant at {self.config.local_path}...")
            client = QdrantClient(path=self.config.local_path) # Initialize a local Qdrant instance that stores data on disk at the specified path (data will persist across runs)
            print(f"\n[OK] Local Qdrant initialized")

        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'docker', 'memory', or 'local'")
        
        return client
    #-------------------
    def create_collection(self, collection_name: str = None, force_recreate: bool = False) -> None:
        """
        Create a Qdrant collection with configured vectors and parameters.
        
        Args:
            collection_name: Name of the collection. Uses config name if None.
            force_recreate: Delete and recreate if collection exists
        """
        # Use collection name from parameter or config
        if collection_name is None:
            collection_name = self.config.collection_name
        
        try:
            self.client.get_collection(collection_name) # Check if collection already exists
            if force_recreate: # If it exists and force_recreate is True, then we delete the existing collection to start fresh
                print(f"Deleting existing collection '{collection_name}'...")
                self.client.delete_collection(collection_name) # Delete the existing collection
            else:
                print(f"Collection '{collection_name}' already exists.")
                return
        except:
            pass
        
        vectors_config = self._build_vectors_config() # Build the vectors configuration based on whether we are in single-vector or multi-vector mode using the helper function defined earlier
        
        # Build HNSW config if provided
        hnsw_config = None
        if self.config.hnsw_config:
            hnsw_config = HnswConfigDiff(**self.config.hnsw_config)
        
        # Create the collection with the specified name, vectors configuration, and HNSW configuration (if provided)
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
            hnsw_config=hnsw_config
        )
        print(f"✓ Collection '{collection_name}' created")

        if self.config.vectors_config:
            print(f"  Mode: Multi-vector with {len(self.config.vectors_config)} vector fields")
            
        else:
            print(f"  Mode: Single-vector ({self.config.embedding_dimension}D, {self.config.distance})")
    #-------------------
    def delete_collection(self, collection_name: str = None) -> None:
        """Delete a collection."""
        if collection_name is None:
            collection_name = self.config.collection_name # Use collection name from parameter or config
        
        try:
            self.client.delete_collection(collection_name) # Delete the specified collection from Qdrant
            print(f"✓ Collection '{collection_name}' deleted")
        except Exception as e:
            print(f"Error deleting collection: {e}")
    #-------------------
    def index_documents(self,
                       documents: List[Dict],
                       collection_name: str = None,
                       batch_size: int = 100) -> int:
        """
        Index documents with pre-computed embeddings into the collection.
        
        For single-vector mode, each document should have:
        - 'id': unique identifier
        - 'embedding': pre-computed vector (np.ndarray or list)
        - 'text': optional, original text for reference
        - 'metadata': optional, additional fields
        
        For multi-vector mode, each document should have:
        - 'id': unique identifier
        - 'embeddings': dict with vector names as keys -> {vector_name: embedding}
        - 'text': optional, original text for reference
        - 'metadata': optional, additional fields
        
        Args:
            documents: List of documents with embeddings
            collection_name: Name of the collection. Uses config name if None.
            batch_size: Number of documents per batch
            
        Returns:
            int: Total number of documents indexed
        """
        if collection_name is None:
            collection_name = self.config.collection_name # Use collection name from parameter or config
        
        points = [] # List to hold PointStruct objects for batch indexing
        total_indexed = 0 # Counter for total indexed documents
        
        # Loop through each document and convert it into a PointStruct for indexing
        for idx, doc in enumerate(documents):
            doc_id = doc.get('id', idx) # Use provided 'id' or fallback to index if not provided
            
            # Determine if single or multi-vector mode
            if self.config.vectors_config:
                # Multi-vector mode
                if 'embeddings' not in doc or not isinstance(doc['embeddings'], dict):
                    print(f"Warning: Document {idx} missing 'embeddings' dict, skipping...")
                    continue
                
                # Convert embeddings dict
                vectors = {}
                for vector_name, embedding in doc['embeddings'].items(): # Loop through each vector field in the embeddings dict
                    if isinstance(embedding, np.ndarray): # Convert numpy array to list if needed for JSON serialization
                        vectors[vector_name] = embedding.tolist()
                    else: # Assume it's already a list
                        vectors[vector_name] = embedding
            else:
                # Single-vector mode
                if 'embedding' not in doc:
                    print(f"Warning: Document {idx} missing 'embedding', skipping...")
                    continue
                
                embedding = doc['embedding'] # Get the embedding for single-vector mode
                if isinstance(embedding, np.ndarray):
                    vectors = embedding.tolist()
                else:
                    vectors = embedding
            
            point = PointStruct( # Create a PointStruct object for this document with the appropriate id, vector(s), and payload (text + metadata)
                id=doc_id, # Use the provided document ID or fallback to the index if not provided
                vector=vectors, # Set the vector field(s) based on whether we are in single-vector or multi-vector mode
                payload={ # The payload can include the original text and any additional metadata fields for reference during retrieval
                    'text': doc.get('text', ''),
                    **doc.get('metadata', {})
                }
            )
            points.append(point) # Add the PointStruct to the batch list
            
            if (idx + 1) % batch_size == 0: # If we've reached the batch size, we upsert the batch of points to Qdrant to index them
                try:
                    self.client.upsert( # Upsert the batch of points to the specified collection in Qdrant, which will add them to the index
                        collection_name=collection_name, # Use the specified collection name
                        points=points # The list of PointStruct objects to index in this batch
                    )
                    total_indexed += len(points) # Update the total indexed count by the number of points in this batch
                    print(f"Indexed {total_indexed} documents...")
                    points = [] # Clear the batch list after indexing
                except Exception as e:
                    print(f"Error upserting batch: {e}")
        
        if points: # If there are any remaining points after the loop that haven't been indexed yet (because they didn't fill a complete batch), we need to index them as well
            try:
                self.client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                total_indexed += len(points)
            except Exception as e:
                print(f"Error upserting final batch: {e}")
        
        print(f"✓ Total {total_indexed} documents indexed")
        return total_indexed
    #-------------------
    def retrieve(self,
                query_embedding,
                collection_name: str = None,
                top_k: int = None,
                query_vector_name: str = None) -> List[Dict]:
        """
        Retrieve the top-k most similar documents using a pre-computed query embedding.
        
        Args:
            query_embedding: Pre-computed query embedding (np.ndarray or list)
            collection_name: Name of the collection. Uses config name if None.
            top_k: Number of results. Uses config value if None.
            query_vector_name: For multi-vector mode, which vector field to search
            
        Returns:
            List[Dict]: Retrieved documents with scores and metadata
        """
        if collection_name is None:
            collection_name = self.config.collection_name # Use collection name from parameter or config
        if top_k is None:
            top_k = self.config.top_k # Use top_k from parameter or config
        
        # Convert numpy array to list if needed
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        
        # For multi-vector collections, specify which vector field to use
        vector_field = None
        if self.config.vectors_config:
            vector_field = query_vector_name or list(self.config.vectors_config.keys())[0] # Default to the first vector field if not specified
        
        try:
            search_results = self.client.query_points( # Query the Qdrant collection using the pre-computed query embedding to retrieve the most similar documents based on the specified vector field (for multi-vector) and top_k results, along with any additional search parameters from the config
                collection_name=collection_name,
                query=query_embedding,
                using=vector_field,
                limit=top_k,
                search_params=self.config.search_params
            ).points
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []
        
        retrieved_docs = [] # List to hold the retrieved documents with their scores and metadata for downstream processing (e.g., RAG pipelines)
        
        # Loop through each result returned from the Qdrant query and extract the relevant information to build a list of retrieved documents with their scores and metadata for use in downstream processing (e.g., RAG pipelines)
        for result in search_results:
            retrieved_docs.append({
                'id': result.id,
                'score': result.score,
                'payload': result.payload,
                'text': result.payload.get('text', '')
            })
        
        return retrieved_docs
    #-------------------
    def retrieve_by_ids(self,
                       doc_ids: List[int],
                       collection_name: str = None) -> List[Dict]:
        """
        Retrieve documents by their IDs.
        
        Args:
            doc_ids: List of document IDs
            collection_name: Name of the collection. Uses config name if None.
            
        Returns:
            List[Dict]: Retrieved documents
        """
        if collection_name is None:
            collection_name = self.config.collection_name
        
        try:
            points = self.client.retrieve( # Retrieve documents from Qdrant by their IDs for the specified collection
                collection_name=collection_name, # Use the specified collection name
                ids=doc_ids # The list of document IDs to retrieve from Qdrant
            )
            
            docs = [] # List to hold the retrieved documents with their metadata for downstream processing (e.g., RAG pipelines)
            for point in points: # Loop through each retrieved point and extract the relevant information to build a list of documents with their metadata for use in downstream processing (e.g., RAG pipelines)
                docs.append({
                    'id': point.id, # The unique identifier of the retrieved document
                    'payload': point.payload, # The payload of the retrieved document, which may include metadata fields
                    'text': point.payload.get('text', '') # The original text of the retrieved document (if stored in the payload) for reference during generation
                })
            return docs
        except Exception as e:
            print(f"Error retrieving by IDs: {e}")
            return []
    #-------------------
    def count_documents(self, collection_name: str = None) -> int:
        """Count documents in a collection."""
        if collection_name is None:
            collection_name = self.config.collection_name
        
        try:
            info = self.client.get_collection(collection_name) # Get the collection information from Qdrant, which includes the count of indexed documents (points) in the collection for reference and monitoring purposes
            return info.points_count if info else 0
        except:
            return 0
    #-------------------
    def get_collection_info(self, collection_name: str = None) -> Dict:
        """Get collection information."""
        if collection_name is None:
            collection_name = self.config.collection_name
        
        try:
            return self.client.get_collection(collection_name)
        except Exception as e:
            print(f"Error retrieving collection info: {e}")
            return {}
    #-------------------
    def clear_collection(self, collection_name: str = None) -> None:
        """Clear all documents from a collection."""
        if collection_name is None:
            collection_name = self.config.collection_name
        
        self.delete_collection(collection_name) # Delete the existing collection to clear all documents
        self.create_collection(collection_name) # Recreate the collection to start fresh after clearing
        print(f"✓ Collection '{collection_name}' cleared")
