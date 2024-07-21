import numpy as NP
from .. import graphUtils as GU
import pytest

def test_generate_triangles_valid(example_ids):
    baseid = 1
    triads = GU.generate_triangles(example_ids, baseid)
    assert len(triads) == (len(example_ids)-1)*(len(example_ids)-2)//2
    unique_ids = NP.unique(example_ids)
    rest_ids = list(unique_ids)
    rest_ids.remove(baseid)
    for e2i in range(len(rest_ids)):
        for e3i in range(e2i+1, len(rest_ids)):
            assert (baseid, rest_ids[e2i], rest_ids[e3i]) in triads

def test_generate_triangles_valid_strings(example_ids_strings):
    baseid = 'A'
    triads = GU.generate_triangles(example_ids_strings, baseid)
    assert len(triads) == (len(example_ids_strings)-1)*(len(example_ids_strings)-2)//2
    unique_ids = NP.unique(example_ids_strings)
    rest_ids = list(unique_ids)
    rest_ids.remove(baseid)
    for e2i in range(len(rest_ids)):
        for e3i in range(e2i+1, len(rest_ids)):
            assert (baseid, rest_ids[e2i], rest_ids[e3i]) in triads

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

