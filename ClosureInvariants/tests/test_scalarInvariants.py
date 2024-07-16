import pytest
import numpy as NP
from .. import scalarInvariants as SI

def test_copol_corrs_list_on_loops_shape(copol_correlations_array, antenna_pairs, loops):
    result = SI.corrs_list_on_loops(copol_correlations_array, antenna_pairs, loops)
    assert len(result) == len(loops)
    for corr_loop in result:
        assert len(corr_loop) == len(loops[0])

# def test_copol_corrs_list_on_loops_type(copol_correlations_array, antenna_pairs, loops):
#     result = SI.corrs_list_on_loops(copol_correlations_array, antenna_pairs, loops)
#     for corr_loop in result:
#         for corr in corr_loop:
#             assert isinstance(corr, NP.ndarray)
