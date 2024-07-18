import pytest
import numpy as NP
from .. import scalarInvariants as SI

def test_copol_corrs_list_on_loops_shape(copol_correlations_array, antenna_pairs, loops):
    result = SI.corrs_list_on_loops(copol_correlations_array, antenna_pairs, loops)
    assert len(result) == len(loops)
    for corr_loop in result:
        assert len(corr_loop) == len(loops[0])

def test_copol_advariant(copol_corrs_list1, copol_advariant1):
    # Call the function
    result = SI.advariant(copol_corrs_list1)

    # Check if the result is a numpy array
    assert isinstance(result, NP.ndarray)    

    # Check if the result has the expected shape
    assert result.shape == copol_advariant1.shape

    # Check if the values are as expected
    assert NP.allclose(result, copol_advariant1)

def test_copol_advariants_multiple_loops(copol_corrs_lol, copol_advariants_on_list):
    # Call the function
    result = SI.advariants_multiple_loops(copol_corrs_lol)

    # Check if the result is a numpy array
    assert isinstance(result, NP.ndarray)    
    
    # Check if the result has the expected shape
    assert result.shape == copol_advariants_on_list.shape

    # Check if the values are as expected
    assert NP.allclose(result, copol_advariants_on_list)

@pytest.mark.parametrize(
    "advariants, normaxis, normwts, normpower, expected_shape",
    [
        (NP.random.randn(3,4,5) + 1j*NP.random.randn(3,4,5), 1, None, 2, (3, 4, 10)),
        (NP.random.randn(3,4,5) + 1j*NP.random.randn(3,4,5), 1, NP.random.rand(3,4,5), 2, (3,4,10)),
        (NP.random.randn(3,4,5) + 1j*NP.random.randn(3,4,5), 1, NP.random.rand(3,4,5), 1, (3,4,10)),
        (NP.random.randn(3,4,5) + 1j*NP.random.randn(3,4,5), 1, NP.zeros((3,4,5)), 2, (3,4,10)),
    ]
)
def test_invariants_from_advariants_method1(advariants, normaxis, normwts, normpower, expected_shape):
    if normwts is not None and NP.all(normwts == 0):
        normwts[:, 0, :] = 1  # For the specific test case where normwts is 1 for the first element and 0 for the rest

    invariants = SI.invariants_from_advariants_method1(advariants, normaxis, normwts=normwts, normpower=normpower)
    assert invariants.shape == expected_shape
    assert NP.all(NP.isfinite(invariants))

@pytest.mark.parametrize(
    "advariants, normaxis, normwts, normpower, exception_type",
    [
        ([1, 2, 3], 0, None, 2, TypeError),
        (NP.random.randn(3,4,5) + 1j * NP.random.randn(3,4,5), 1, [1,2,3], 2, TypeError),
        (NP.random.randn(3,4,5) + 1j * NP.random.randn(3,4,5), 1, NP.random.rand(3,4,5), '2', TypeError),
    ]
)
def test_invariants_from_advariants_method1_exceptions(advariants, normaxis, normwts, normpower, exception_type):
    with pytest.raises(exception_type):
        SI.invariants_from_advariants_method1(advariants, normaxis, normwts=normwts, normpower=normpower)
