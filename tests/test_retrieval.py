from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project import HybridRetriever, load_documents, reciprocal_rank_fusion, run


class OfflineClient:
    enabled = False


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.retriever = HybridRetriever(load_documents())

    def test_pump_question_has_pump_evidence(self):
        result = self.retriever.search("P-101 泵振动排查")
        self.assertEqual(result[0]["id"], "D01")

    def test_unknown_query_abstains(self):
        self.assertEqual(self.retriever.search("zxqv987654 galactic nebula"), [])

    def test_empty_query_is_rejected(self):
        with self.assertRaises(ValueError):
            self.retriever.search("  ")

    def test_fusion_rewards_agreement(self):
        scores = reciprocal_rank_fusion([[0, 1], [1, 2]])
        self.assertGreater(scores[1], scores[0])

    def test_offline_result_includes_real_citations(self):
        result = run({"text": "P-101 轴承振动"}, OfflineClient())
        self.assertIn("[D01]", result["summary"])


if __name__ == "__main__":
    unittest.main()
