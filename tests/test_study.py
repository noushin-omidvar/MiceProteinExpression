import pandas as pd
from study import marginalize


def test_factor_marginal_sums_group_probabilities_by_label():
    joint = pd.DataFrame({"c-CS-m": [0.3], "t-CS-s": [0.2], "c-SC-s": [0.5]}, index=["309"])
    mapping = pd.Series({"c-SC-s": "S/C", "c-CS-m": "C/S", "t-CS-s": "C/S"})
    expected = pd.DataFrame({"C/S": [0.5], "S/C": [0.5]}, index=["309"])
    pd.testing.assert_frame_equal(marginalize(joint, mapping), expected)
