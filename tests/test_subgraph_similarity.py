from __future__ import annotations

import unittest

from src.eval.subgraph_similarity import cosine_embedding_similarity


class SubgraphSimilarityTests(unittest.TestCase):
    def test_cosine_embedding_similarity_identical(self) -> None:
        self.assertEqual(cosine_embedding_similarity([1, 0], [1, 0]), 1.0)

    def test_cosine_embedding_similarity_orthogonal(self) -> None:
        self.assertEqual(cosine_embedding_similarity([1, 0], [0, 1]), 0.0)


if __name__ == "__main__":
    unittest.main()
