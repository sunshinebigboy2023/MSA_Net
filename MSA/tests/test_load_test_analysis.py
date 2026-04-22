import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.load_test_analysis import Sample, percentile, summarize


class LoadTestAnalysisTests(unittest.TestCase):
    def test_summarize_reports_success_rate_and_latency(self):
        report = summarize(
            [
                Sample(True, 200, 10, task_id="a"),
                Sample(True, 200, 20, task_id="b"),
                Sample(False, 429, 30, error="limited"),
            ]
        )

        self.assertEqual(report["total"], 3)
        self.assertEqual(report["ok"], 2)
        self.assertEqual(report["failed"], 1)
        self.assertEqual(report["successRate"], 0.6667)
        self.assertEqual(report["statusCounts"], {200: 2, 429: 1})
        self.assertEqual(report["errorCounts"], {"limited": 1})

    def test_percentile_handles_empty_and_sorted_values(self):
        self.assertEqual(percentile([], 95), 0)
        self.assertEqual(percentile([10, 20, 30, 40], 95), 40)


if __name__ == "__main__":
    unittest.main()
