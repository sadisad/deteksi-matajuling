"""Tests for the decision layer.

Run with:  python -m unittest discover -s tests
They need no model weights, no camera, and no GPU.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from screening import (  # noqa: E402
    INCONCLUSIVE,
    NORMAL,
    STRABISMUS,
    Detection,
    Outcome,
    decide,
    format_result,
)


class NoOutputIsNotNormal(unittest.TestCase):
    def test_empty_detections_abstain(self):
        outcome = decide([])
        self.assertEqual(outcome.status, INCONCLUSIVE)
        self.assertIsNone(outcome.confidence)

    def test_abstention_never_invents_a_score(self):
        outcome = decide([])
        self.assertNotEqual(outcome.confidence, 1.0)
        self.assertNotIn("%", format_result(outcome))

    def test_missing_eye_regions_abstain_even_with_detections(self):
        outcome = decide([Detection("NORMAL", 0.93)], eye_regions_found=False)
        self.assertEqual(outcome.status, INCONCLUSIVE)
        self.assertIsNone(outcome.confidence)
        self.assertIn("mata", outcome.reason)


class ConclusivePaths(unittest.TestCase):
    def test_clean_normal_report_keeps_its_confidence(self):
        outcome = decide([Detection("NORMAL", 0.87)])
        conf = outcome.confidence
        assert conf is not None
        self.assertEqual(outcome.status, NORMAL)
        self.assertAlmostEqual(conf, 0.87)
        self.assertEqual(format_result(outcome), "NORMAL (87.00%)")

    def test_single_eye_is_enough_for_strabismus(self):
        outcome = decide([Detection("NORMAL", 0.95), Detection("JULING", 0.61)])
        conf = outcome.confidence
        assert conf is not None
        self.assertEqual(outcome.status, STRABISMUS)
        self.assertAlmostEqual(conf, 0.61)

    def test_strabismus_confidence_ignores_the_normal_box(self):
        outcome = decide([Detection("NORMAL", 0.99), Detection("STRABISMUS", 0.42)])
        conf = outcome.confidence
        assert conf is not None
        self.assertAlmostEqual(conf, 0.42)

    def test_lowercase_labels_still_match(self):
        outcome = decide([Detection("juling", 0.55)])
        self.assertEqual(outcome.status, STRABISMUS)


class Formatting(unittest.TestCase):
    def test_inconclusive_prints_no_percentage(self):
        self.assertEqual(format_result(Outcome(INCONCLUSIVE, None, "x")), INCONCLUSIVE)

    def test_conclusive_prints_percentage(self):
        self.assertEqual(format_result(Outcome(NORMAL, 0.5)), "NORMAL (50.00%)")


if __name__ == "__main__":
    unittest.main()
