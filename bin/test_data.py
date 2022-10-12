import pandas as pd
import unittest
import os

ROOTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATADIR = os.path.join(ROOTDIR, "data")


class TestData(unittest.TestCase):
    """Tests for data script."""

    def setUp(self):
        self.data = pd.read_csv(os.path.join(DATADIR, "data.csv"))

    def test_adult_sex_examples_count(self):
        """Test examples for adult dataset.

        Test 1:
            The adult dataset must contains 9 feature sets.

        Test 2:
            Each feature set must contain metrics for 3 model variants (2
            models plus the data).

        Test 3:
            Each model variant must contain metrics for 3 privileged
            group variants (privileged group, unprivileged group
            & unconditioned).

        Test 4:
            The adult dataset must contain a total of 9*3*3 = 81 examples.

        """
        adult = self.data[self.data["dataset_label"] == "adult"]
        num_featues = adult["num_features"].unique().tolist()
        models = adult["model"].unique().tolist()

        # Test 1:
        self.assertEqual(adult["num_features"].value_counts().shape[0], 9)

        # Test 2:
        for n in num_featues:
            self.assertEqual(
                adult[adult["num_features"] == n]["model"].value_counts().shape[0], 3
            )

        # Test 3:
        for model in models:
            self.assertEqual(
                adult[adult["model"] == model]["privileged"].value_counts().shape[0], 3
            )

        # Test 4:
        self.assertEqual(adult[self.data["dataset_label"] == "adult"].shape[0], 81)
