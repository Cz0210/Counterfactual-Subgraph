import tempfile
import unittest
from pathlib import Path

from scripts.eval.collect_gcf_official_results import collect_summary, parse_summary_log_text


class CollectGCFOfficialResultsTests(unittest.TestCase):
    def test_parses_official_two_section_top_lines(self) -> None:
        content = """Evaluation Coverage,  Threshold: 0.1 running...
Top 1: 0.25
Top 5: 0.50
Top 10: 0.75
Calculating cost...
Top 1: 0.90
Top 5: 0.40
Top 10: 0.20
"""
        rows, message = parse_summary_log_text(content)

        self.assertEqual(message, "")
        self.assertEqual(len(rows), 3)
        top10 = next(row for row in rows if row["k"] == 10)
        self.assertAlmostEqual(top10["coverage"], 0.75)
        self.assertAlmostEqual(top10["cost"], 0.20)

    def test_unknown_format_writes_parse_failure_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "summary.log"
            log_path.write_text("no metrics here\n", encoding="utf-8")

            payload = collect_summary(log_path)

        self.assertFalse(payload["parse_ok"])
        self.assertEqual(payload["raw_summary_log_path"], str(log_path))
        self.assertIn("No coverage/cost metrics", payload["error_message"])

    def test_parses_explicit_coverage_cost_line(self) -> None:
        rows, message = parse_summary_log_text("top 10 coverage 0.66 cost 0.12\n")

        self.assertEqual(message, "")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["k"], 10)
        self.assertAlmostEqual(rows[0]["coverage"], 0.66)
        self.assertAlmostEqual(rows[0]["cost"], 0.12)


if __name__ == "__main__":
    unittest.main()
