import numpy as NP
from .. import graphUtils as GU
import pytest

def test_generate_triangles_valid(example_ids):
    baseid = 1
    triads = GU.generate_triangles(example_ids, baseid)
    assert len(triads) == 3
    assert (baseid, 2, 3) in triads
    assert (baseid, 2, 4) in triads
    assert (baseid, 3, 4) in triads

def test_generate_triangles_valid_strings(example_ids_strings):
    baseid = 'A'
    triads = GU.generate_triangles(example_ids_strings, baseid)
    assert len(triads) == 3
    assert (baseid, 'B', 'C') in triads
    assert (baseid, 'B', 'D') in triads
    assert (baseid, 'C', 'D') in triads

def test_generate_triangles_invalid_ids():
    with pytest.raises(TypeError):
        GU.generate_triangles(123, 1)

def test_generate_triangles_invalid_baseid(example_ids):
    with pytest.raises(TypeError):
        GU.generate_triangles(example_ids, 1.5)

def test_generate_triangles_invalid_baseid_value(example_ids):
    with pytest.raises(ValueError):
        GU.generate_triangles(example_ids, 5)

def test_generate_triangles_invalid_ids_length():
    with pytest.raises(ValueError):
        GU.generate_triangles([1], 1)

def test_output_shapes_and_values_unpack_realimg_to_real(arrtype, realimag_scalar_array, realimag_matrix_array, real_scalar_array, real_matrix_array):
    if arrtype == 'scalar':
        input_array = realimag_scalar_array
        expected_output = real_scalar_array
    elif arrtype == 'matrix':
        input_array = realimag_matrix_array
        expected_output = real_matrix_array

    result = GU.unpack_realimag_to_real(input_array, arrtype=arrtype)
    assert result.shape == expected_output.shape, "Output shape mismatch for arrtype='{}'".format(arrtype)
    assert NP.allclose(result, expected_output), "Output value mismatch for arrtype='{}'".format(arrtype)

def test_output_shapes_and_values_repack_real_to_realimag(arrtype, real_scalar_array, real_matrix_array, realimag_scalar_array, realimag_matrix_array):
    if arrtype == 'scalar':
        input_array = real_scalar_array
        expected_output = realimag_scalar_array
    elif arrtype == 'matrix':
        input_array = real_matrix_array
        expected_output = realimag_matrix_array

    result = GU.repack_real_to_realimag(input_array, arrtype=arrtype)
    assert result.shape == expected_output.shape, "Output shape mismatch for arrtype='{}'".format(arrtype)
    assert NP.allclose(result, expected_output), "Output value mismatch for arrtype='{}'".format(arrtype)

