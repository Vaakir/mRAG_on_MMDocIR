"""
Qdrant Vector Database Utilities

Generic, reusable Qdrant vector database functions that can be used by any pipeline.
Supports three modes: local (disk), memory (in-memory), and docker (containerized).
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, HnswConfigDiff, Filter, FieldCondition, MatchAny

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
    def __init__(self, config):
        """
        Initialize the vector database.
        
        Args:
            config: BaselineConfig or AdvancedConfig object with database settings
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
        if hasattr(self.config, "VECTOR_DB_VECTORS_CONFIG") and self.config.VECTOR_DB_VECTORS_CONFIG:
            # Multi-vector mode
            vectors_config = {}
            
            # Loop through each vector field in the config and create a VectorParams object for it
            for vector_name, params in self.config.VECTOR_DB_VECTORS_CONFIG.items():
                dimension = params.get("dimension", self.config.EMBEDDING_DIMENSION)           # Default dimension if not specified
                distance_str = params.get("distance", self.config.VECTOR_DB_DISTANCE)    # Default distance if not specified
                distance = self._get_distance_metric(distance_str) # Convert distance string to Distance enum
                # For multi-vector collections, we use a dict of vector_name -> VectorParams
                vectors_config[vector_name] = VectorParams(
                    size=dimension,
                    distance=distance
                )
            return vectors_config
        else:
            # Single-vector mode
            distance = self._get_distance_metric(self.config.VECTOR_DB_DISTANCE) # Convert distance string to Distance enum
            return VectorParams( # For single-vector collections, we return a single VectorParams object
                size=self.config.EMBEDDING_DIMENSION,
                distance=distance
            )
    #-------------------
    def _initialize_client(self) -> QdrantClient:
        """Initialize Qdrant client based on config mode."""
        mode = self.config.VECTOR_DB_MODE.lower() # Get the database mode from the configuration and convert it to lowercase for consistency
        
        if mode == "docker":
            docker_url = self.config.VECTOR_DB_DOCKER_URL
            print(f"Connecting to Qdrant Docker at {docker_url}...")
            client = QdrantClient(url=docker_url) # Connect to the Qdrant instance running in Docker using the specified URL
            print("✓ Connected to Qdrant Docker")

        elif mode == "memory":
            print("Initializing in-memory Qdrant...")
            client = QdrantClient(":memory:") # Initialize an in-memory Qdrant instance (data will not persist after the program ends)
            print("[OK] In-memory Qdrant initialized")

        elif mode == "local":
            local_path = self.config.VECTOR_DB_PATH
            print(f"Initializing local Qdrant at {local_path}...")
            client = QdrantClient(path=local_path) # Initialize a local Qdrant instance that stores data on disk at the specified path (data will persist across runs)
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
        if collection_name is None:
            collection_name = self.config.VECTOR_DB_COLLECTION
        
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
        if hasattr(self.config, "VECTOR_DB_HNSW_CONFIG") and self.config.VECTOR_DB_HNSW_CONFIG:
            hnsw_config = HnswConfigDiff(**self.config.VECTOR_DB_HNSW_CONFIG)
        
        # Create the collection with the specified name, vectors configuration, and HNSW configuration (if provided)
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
            hnsw_config=hnsw_config
        )
        print(f"✓ Collection '{collection_name}' created")

        if hasattr(self.config, "VECTOR_DB_VECTORS_CONFIG") and self.config.VECTOR_DB_VECTORS_CONFIG:
            print(f"  Mode: Multi-vector with {len(self.config.VECTOR_DB_VECTORS_CONFIG)} vector fields")
            
        else:
            print(f"  Mode: Single-vector ({self.config.EMBEDDING_DIMENSION}D, {self.config.VECTOR_DB_DISTANCE})")
    #-------------------
    def delete_collection(self, collection_name: str = None) -> None:
        """Delete a collection."""
        if collection_name is None:
            collection_name = self.config.VECTOR_DB_COLLECTION
        
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
            collection_name = self.config.VECTOR_DB_COLLECTION # Use collection name from parameter or config
        
        points = [] # List to hold PointStruct objects for batch indexing
        total_indexed = 0 # Counter for total indexed documents
        
        # Loop through each document and convert it into a PointStruct for indexing
        for idx, doc in enumerate(documents):
            doc_id = doc.get('id', idx) # Use provided 'id' or fallback to index if not provided
            
            # Determine if single or multi-vector mode
            if hasattr(self.config, "VECTOR_DB_VECTORS_CONFIG") and self.config.VECTOR_DB_VECTORS_CONFIG:
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
                query_vector_name: str = None,
                allowed_types: Optional[List[str]] = None) -> List[Dict]:
        """
        Retrieve the top-k most similar documents using a pre-computed query embedding.

        Args:
            query_embedding: Pre-computed query embedding (np.ndarray or list)
            collection_name: Name of the collection. Uses config name if None.
            top_k: Number of results. Uses config value if None.
            query_vector_name: For multi-vector mode, which vector field to search
            allowed_types: If set, only return chunks whose 'type' payload field is in this list.
                           E.g. ["text", "page_image"] excludes any other types.

        Returns:
            List[Dict]: Retrieved documents with scores and metadata
        """
        if collection_name is None:
            collection_name = self.config.VECTOR_DB_COLLECTION
        if top_k is None:
            top_k = self.config.TOP_K

        # Convert numpy array to list if needed
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()

        # For multi-vector collections, specify which vector field to use
        vector_field = None
        if hasattr(self.config, "VECTOR_DB_VECTORS_CONFIG") and self.config.VECTOR_DB_VECTORS_CONFIG:
            vector_field = query_vector_name or list(self.config.VECTOR_DB_VECTORS_CONFIG.keys())[0] # Default to the first vector field if not specified

        # Build type filter if allowed_types is specified
        query_filter = None
        if allowed_types:
            query_filter = Filter(
                must=[FieldCondition(key="type", match=MatchAny(any=allowed_types))]
            )

        try:
            search_params = self.config.VECTOR_DB_SEARCH_PARAMS if hasattr(self.config, "VECTOR_DB_SEARCH_PARAMS") else None
            search_results = self.client.query_points(
                collection_name=collection_name,
                query=query_embedding,
                using=vector_field,
                limit=top_k,
                query_filter=query_filter,
                search_params=search_params
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
    def retrieve_split(self,
                       query_embedding,
                       top_k_text: int = 3,
                       top_k_image: int = 2,
                       collection_name: str = None) -> List[Dict]:
        """
        Retrieve top-k text chunks and top-k page_image chunks separately, then
        merge sorted by score. Prevents text-text similarity from crowding out
        cross-modal text-image results, which typically score lower in CLIP space.

        Args:
            query_embedding: Pre-computed query embedding
            top_k_text: Number of text chunks to retrieve
            top_k_image: Number of page_image chunks to retrieve
            collection_name: Uses config name if None

        Returns:
            List[Dict]: Merged results sorted by score descending
        """
        text_results  = self.retrieve(query_embedding, collection_name=collection_name,
                                      top_k=top_k_text,  allowed_types=["text"])
        image_results = self.retrieve(query_embedding, collection_name=collection_name,
                                      top_k=top_k_image, allowed_types=["page_image"])
        merged = text_results + image_results
        merged.sort(key=lambda r: r["score"], reverse=True)
        return merged
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
            collection_name = self.config.VECTOR_DB_COLLECTION
        
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
            collection_name = self.config.VECTOR_DB_COLLECTION
        
        try:
            info = self.client.get_collection(collection_name) # Get the collection information from Qdrant, which includes the count of indexed documents (points) in the collection for reference and monitoring purposes
            return info.points_count if info else 0
        except:
            return 0
    #-------------------
    def get_collection_info(self, collection_name: str = None) -> Dict:
        """Get collection information."""
        if collection_name is None:
            collection_name = self.config.VECTOR_DB_COLLECTION
        
        try:
            return self.client.get_collection(collection_name)
        except Exception as e:
            print(f"Error retrieving collection info: {e}")
            return {}
    #-------------------
    def clear_collection(self, collection_name: str = None) -> None:
        """Clear all documents from a collection."""
        if collection_name is None:
            collection_name = self.config.VECTOR_DB_COLLECTION
        
        self.delete_collection(collection_name) # Delete the existing collection to clear all documents
        self.create_collection(collection_name) # Recreate the collection to start fresh after clearing
        print(f"✓ Collection '{collection_name}' cleared")
