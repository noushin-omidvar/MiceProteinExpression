import numpy as np

from mice_protein.statistics import benjamini_hochberg


def test_bh_is_bounded_and_monotone_by_pvalue():
    p = np.array([0.04, 0.001, 0.02, 0.8])
    q = benjamini_hochberg(p)
    assert np.all((q >= 0) & (q <= 1))
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12)
