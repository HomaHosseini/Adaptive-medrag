import numpy as np


class EvidenceTracker:
    """
    Tracks retrieved evidence across iterations.

    Responsibilities:
    - document novelty
    - semantic novelty
    - semantic redundancy
    - novelty history
    - saturation-related statistics

    It does NOT decide STOP / CONTINUE.
    """

    def __init__(self, retrieval_system):
        self.retrieval_system = retrieval_system

        # All document IDs retrieved so far
        self.previous_document_ids = set()

        # All MedCPT embeddings retrieved so far
        self.previous_embeddings = None

        # History for analysis
        self.history = []


    def document_novelty(self, current_ids):

        current_ids = set(
            str(x)
            for x in current_ids
            if x is not None
        )

        if not current_ids:
            return 0.0, 0

        # First iteration
        if not self.previous_document_ids:
            return 1.0, len(current_ids)

        new_ids = (
            current_ids
            - self.previous_document_ids
        )

        novelty = (
            len(new_ids)
            / len(current_ids)
        )

        return novelty, len(new_ids)



    def get_embeddings(self, document_ids):
        """
        Retrieve the existing document embeddings from the
        MedCPT FAISS index.

        Returns:
            embeddings: (N, dim) numpy array
            found_ids: document IDs successfully found
        """

        embeddings, found_ids = (
            self.retrieval_system.get_embeddings(
                document_ids
            )
        )

        if embeddings is None:
            return None, []

        embeddings = np.asarray(embeddings, dtype=np.float32)

        return embeddings, found_ids


    @staticmethod
    def _normalize(embeddings):
        """
        L2-normalize embedding vectors so that dot product
        corresponds to cosine similarity.
        """

        if embeddings is None:
            return None

        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True
        )

        return embeddings / np.maximum(
            norms,
            1e-12
        )

    # ==========================================================
    # 4. SEMANTIC NOVELTY
    # ==========================================================

    def semantic_novelty(
        self,
        current_embeddings
    ):
        """
        Compare current evidence against ALL previously
        retrieved evidence.

        For each current document:

            redundancy =
                max cosine similarity to previous evidence

            novelty =
                1 - redundancy

        Returns a dictionary rather than a single number so
        the controller can later use different statistics.
        """

        if (current_embeddings is None or len(current_embeddings) == 0):
            return None

        if (self.previous_embeddings is None or len(self.previous_embeddings) == 0):
            return None

        current = self._normalize(current_embeddings)

        previous = self._normalize(self.previous_embeddings)

        similarity_matrix = (
            current @ previous.T
        )

        # For every new document, find its closest
        # previously retrieved document.
        max_similarity = similarity_matrix.max(
            axis=1
        )

        # 1 - similarity = semantic novelty
        novelty_values = (
            1.0 - max_similarity
        )

        return {
            "mean": float(
                novelty_values.mean()
            ),

            "median": float(
                np.median(novelty_values)
            ),

            "high_novelty_fraction": float(
                np.mean(
                    novelty_values >= 0.5
                )
            ),

            "mean_redundancy": float(
                max_similarity.mean()
            ),

            "novelty_values": [
                float(x)
                for x in novelty_values
            ]
        }


    def update(self, current_ids):
        """
        Process one retrieval iteration.

        Returns a dictionary containing all evidence statistics.
        """

        # ========================================================
        # 1. CLEAN / DEDUPLICATE IDS
        # ========================================================

        current_ids = list(
            dict.fromkeys(
                str(x)
                for x in current_ids
                if x is not None
            )
        )

        # ========================================================
        # 2. DOCUMENT NOVELTY
        # ========================================================

        document_novelty, num_new_documents = (
            self.document_novelty(
                current_ids
            )
        )

        # ========================================================
        # 3. MEDCPT EMBEDDINGS
        # ========================================================

        current_embeddings, found_ids = (
            self.get_embeddings(
                current_ids
            )
        )

        # ========================================================
        # 4. SEMANTIC NOVELTY
        # ========================================================

        semantic_stats = (
            self.semantic_novelty(
                current_embeddings
            )
        )

        # ========================================================
        # 5. SAVE RESULT
        # ========================================================

        result = {
            "num_documents":
                len(current_ids),

            "num_new_documents":
                num_new_documents,

            "document_novelty":
                document_novelty,

            "semantic_novelty": (
                None
                if semantic_stats is None
                else semantic_stats["mean"]
            ),

            "semantic_median_novelty": (
                None
                if semantic_stats is None
                else semantic_stats["median"]
            ),

            "high_semantic_novelty_fraction": (
                None
                if semantic_stats is None
                else semantic_stats[
                    "high_novelty_fraction"
                ]
            ),

            "semantic_redundancy": (
                None
                if semantic_stats is None
                else semantic_stats[
                    "mean_redundancy"
                ]
            ),

            "novelty_values": (
                None
                if semantic_stats is None
                else semantic_stats[
                    "novelty_values"
                ]
            ),

            "found_embeddings":
                len(found_ids),

            "novelty_available":
                semantic_stats is not None
        }

        # ========================================================
        # 6. SAVE HISTORY
        # ========================================================

        self.history.append(
            result
        )

        # ========================================================
        # 7. UPDATE CUMULATIVE DOCUMENT IDS
        # ========================================================

        self.previous_document_ids.update(
            current_ids
        )

        # ========================================================
        # 8. UPDATE CUMULATIVE EMBEDDINGS
        # ========================================================

        if current_embeddings is not None:

            current_embeddings = np.asarray(
                current_embeddings,
                dtype=np.float32
            )

            if self.previous_embeddings is None:
                self.previous_embeddings = (
                    current_embeddings
                )

            else:
                self.previous_embeddings = (
                    np.vstack([
                        self.previous_embeddings,
                        current_embeddings
                    ])
                )

        return result

        
    def reset(self):
        """
        Reset tracker state for a new question / trajectory.
        """

        self.previous_document_ids = set()
        self.previous_embeddings = None
        self.history = []