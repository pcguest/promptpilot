import os
import shutil
import faiss
import numpy as np
import json

from .embedder import Embedder

class Brain:
    """
    Manages per-project memory using FAISS for similarity search and local storage.

    This class provides functionalities to add text memories, search for similar
    memories, and delete all memories associated with a specific project.
    It uses an Embedder to generate text embeddings and FAISS for efficient
    similarity searching. Raw text data is stored alongside the FAISS index.
    """

    def __init__(self, project_id: str, store_base_path: str = "brain/store"):
        """
        Initialises the Brain for a specific project.

        Args:
            project_id: A unique identifier for the project.
            store_base_path: The base directory path where project memories are stored.
                             Defaults to "brain/store".

        Raises:
            ValueError: If project_id is empty or not a string.
        """
        if not project_id or not isinstance(project_id, str):
            raise ValueError("project_id must be a non-empty string.")

        self.project_id = project_id
        self.embedder = Embedder() # Assumes Embedder can be initialised without arguments here
                                   # or that OPENAI_API_KEY is globally available via .env

        self.project_path = os.path.join(store_base_path, self.project_id)
        self.index_file = os.path.join(self.project_path, "index.faiss")
        self.text_data_file = os.path.join(self.project_path, "text_data.json")

        self._ensure_project_path_exists()
        self._load_memory()

    def _ensure_project_path_exists(self):
        """Ensures the project-specific storage directory exists."""
        os.makedirs(self.project_path, exist_ok=True)

    def _load_memory(self):
        """
        Loads the FAISS index and text data from disk if they exist.
        If not, initialises an empty index and text data store.
        """
        self.texts: list[str] = []
        self.ids: list[int] = [] # Stores original index of text in self.texts

        if os.path.exists(self.index_file) and os.path.exists(self.text_data_file):
            try:
                self.index = faiss.read_index(self.index_file)
                with open(self.text_data_file, 'r', encoding='utf-8') as f:
                    stored_data = json.load(f)
                    self.texts = stored_data.get('texts', [])
                    self.ids = stored_data.get('ids', list(range(len(self.texts))))

                # Ensure index dimension matches embedder output if index is not empty
                if self.index.ntotal > 0:
                    embedding_dim = self.embedder.get_embedding("test").shape[0] if hasattr(self.embedder.get_embedding("test"), 'shape') else len(self.embedder.get_embedding("test"))
                    if self.index.d != embedding_dim:
                        print(f"Warning: Index dimension ({self.index.d}) does not match embedder dimension ({embedding_dim}). Re-initialising index.")
                        self._initialise_empty_index()
            except Exception as e:
                print(f"Error loading memory for project {self.project_id}: {e}. Re-initialising memory.")
                self._initialise_empty_index()
        else:
            self._initialise_empty_index()

    def _initialise_empty_index(self):
        """Initialises an empty FAISS index and text data store."""
        # Determine embedding dimension from the embedder
        # This requires generating a dummy embedding if not known beforehand.
        try:
            # Attempt to get embedding for a non-empty string
            sample_embedding = self.embedder.get_embedding("initial_dimension_check")
            # Check if it's a numpy array or list to get dimension
            dimension = sample_embedding.shape[0] if hasattr(sample_embedding, 'shape') else len(sample_embedding)
        except Exception as e:
            print(f"Could not determine embedding dimension: {e}. Defaulting to 1536 (text-embedding-ada-002).")
            dimension = 1536 # Default for "text-embedding-ada-002"

        self.index = faiss.IndexFlatL2(dimension) # L2 distance for similarity
        self.index = faiss.IndexIDMap(self.index) # Allows mapping to original IDs
        self.texts = []
        self.ids = []


    def _save_memory(self):
        """Saves the FAISS index and text data to disk."""
        if self.index is None:
            print("Error: Index is not initialised. Cannot save memory.")
            return

        faiss.write_index(self.index, self.index_file)
        with open(self.text_data_file, 'w', encoding='utf-8') as f:
            json.dump({'texts': self.texts, 'ids': self.ids}, f)

    def add_memory(self, text: str):
        """
        Adds a text to the project's memory.

        The text is embedded, and both the embedding and the original text are stored.

        Args:
            text: The text string to add to memory.

        Raises:
            ValueError: If the text is empty or not a string.
        """
        if not text or not isinstance(text, str):
            raise ValueError("Memory text must be a non-empty string.")

        embedding = self.embedder.get_embedding(text)
        embedding_np = np.array([embedding], dtype=np.float32)

        new_id = len(self.texts) # Use the next available integer ID

        self.index.add_with_ids(embedding_np, np.array([new_id], dtype=np.int64))
        self.texts.append(text)
        self.ids.append(new_id) # Store the ID used for this text
        self._save_memory()
        print(f"Memory added for project {self.project_id}. Total memories: {len(self.texts)}")


    def search_memory(self, query: str, k: int = 5) -> list[str]:
        """
        Searches the project's memory for texts similar to the query.

        Args:
            query: The query string.
            k: The number of similar texts to retrieve. Defaults to 5.

        Returns:
            A list of the k most similar text strings found in memory.
            Returns an empty list if no memories are present or no matches are found.

        Raises:
            ValueError: If the query is empty or not a string.
        """
        if not query or not isinstance(query, str):
            raise ValueError("Search query must be a non-empty string.")

        if self.index.ntotal == 0:
            return [] # No memories to search

        query_embedding = self.embedder.get_embedding(query)
        query_embedding_np = np.array([query_embedding], dtype=np.float32)

        # Search the index
        distances, indices = self.index.search(query_embedding_np, k=min(k, self.index.ntotal))

        results = []
        if indices.size > 0:
            for i in range(indices.shape[1]):
                idx = indices[0, i]
                if idx != -1 : # FAISS returns -1 for no item/padding
                    # Find the original text corresponding to this index ID
                    original_text_index = self.ids.index(idx) # Find position of this ID in self.ids
                    results.append(self.texts[original_text_index])
        return results

    def delete_memory(self):
        """
        Deletes all memory associated with the current project.

        This includes the FAISS index file and the text data file.
        The project-specific directory within `brain/store/` is removed.
        """
        if os.path.exists(self.project_path):
            try:
                shutil.rmtree(self.project_path)
                print(f"All memory for project {self.project_id} has been deleted from {self.project_path}.")
            except OSError as e:
                print(f"Error deleting memory for project {self.project_id}: {e}")
                # Potentially raise an error or handle more gracefully
        else:
            print(f"No memory found for project {self.project_id} at {self.project_path}. Nothing to delete.")

        # Re-initialise to an empty state in memory as well
        self._initialise_empty_index()

if __name__ == '__main__':
    # This example requires a .env file with OPENAI_API_KEY in the root directory
    # and the necessary libraries (openai, faiss-cpu, numpy, python-dotenv) installed.
    print("Running Brain class example...")
    example_project_id = "test_project_123"

    # Clean up any previous test run
    print(f"Attempting to clean up previous run for project: {example_project_id}")
    temp_brain_for_cleanup = Brain(example_project_id)
    temp_brain_for_cleanup.delete_memory() # Ensures clean state for test
    del temp_brain_for_cleanup
    print("-" * 30)

    try:
        # 1. Initialise Brain
        print(f"1. Initialising Brain for project: {example_project_id}")
        brain_instance = Brain(project_id=example_project_id)
        print(f"Project path: {brain_instance.project_path}")
        print(f"Initial memory count: {brain_instance.index.ntotal}")
        print("-" * 30)

        # 2. Add memories
        print("2. Adding memories...")
        memories_to_add = [
            "The first piece of information is about apples.",
            "The second memory talks about bananas and their colour.",
            "Oranges are citrus fruits, rich in Vitamin C.",
            "Prompt engineering is key to good LLM outputs.",
            "The sky is often blue during the day."
        ]
        for mem in memories_to_add:
            brain_instance.add_memory(mem)
        print(f"Memory count after adding: {brain_instance.index.ntotal}")
        assert len(brain_instance.texts) == len(memories_to_add)
        print("-" * 30)

        # 3. Search memory
        print("3. Searching memory...")
        search_query_1 = "fruits"
        results_1 = brain_instance.search_memory(search_query_1, k=3)
        print(f"Search results for '{search_query_1}' (k=3):")
        for res in results_1:
            print(f"  - {res}")
        assert len(results_1) <= 3
        print("-" * 15)

        search_query_2 = "what colour is the sky?"
        results_2 = brain_instance.search_memory(search_query_2, k=2)
        print(f"Search results for '{search_query_2}' (k=2):")
        for res in results_2:
            print(f"  - {res}")
        assert len(results_2) <= 2
        print("-" * 30)

        # 4. Test persistence: Load a new Brain instance for the same project
        print("4. Testing persistence by re-loading Brain...")
        brain_instance_reloaded = Brain(project_id=example_project_id)
        print(f"Reloaded memory count: {brain_instance_reloaded.index.ntotal}")
        assert brain_instance_reloaded.index.ntotal == len(memories_to_add)

        search_query_3 = "LLMs"
        results_3 = brain_instance_reloaded.search_memory(search_query_3, k=1)
        print(f"Search results for '{search_query_3}' (k=1) using reloaded Brain:")
        for res in results_3:
            print(f"  - {res}")
        assert "Prompt engineering" in results_3[0] if results_3 else False
        print("-" * 30)

        # 5. Delete memory
        print("5. Deleting memory...")
        brain_instance.delete_memory()
        print(f"Memory count after deletion: {brain_instance.index.ntotal}")
        assert brain_instance.index.ntotal == 0
        assert not os.path.exists(brain_instance.project_path)
        print("-" * 30)

        # 6. Verify deletion by trying to load again
        print("6. Verifying deletion by attempting to load again...")
        brain_instance_after_delete = Brain(project_id=example_project_id)
        print(f"Memory count after re-initialising post-delete: {brain_instance_after_delete.index.ntotal}")
        assert brain_instance_after_delete.index.ntotal == 0

        # Test adding memory again after deletion
        brain_instance_after_delete.add_memory("A new memory after deletion.")
        print(f"Memory count after adding new data post-delete: {brain_instance_after_delete.index.ntotal}")
        assert brain_instance_after_delete.index.ntotal == 1

        # Final cleanup for the test project
        brain_instance_after_delete.delete_memory()


        print("\nBrain class example completed successfully.")

    except openai.APIError as apie:
        print(f"\nOpenAI API Error during example: {apie}")
        print("Please ensure your OPENAI_API_KEY is correctly set in a .env file and is valid.")
    except ImportError as ie:
        print(f"\nImport Error: {ie}. Please ensure faiss-cpu, openai, numpy, and python-dotenv are installed.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during the Brain class example: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Final cleanup attempt, in case of errors partway through
        if os.path.exists(os.path.join("brain/store", example_project_id)):
            print(f"Final cleanup: Removing test project directory {os.path.join('brain/store', example_project_id)}")
            shutil.rmtree(os.path.join("brain/store", example_project_id))
