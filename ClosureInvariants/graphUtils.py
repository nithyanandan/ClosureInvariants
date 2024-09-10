import itertools
from typing import List, Tuple, Union

import numpy as NP

def generate_triads(ids, baseid):
    """
    Generate triads of IDs based on a base ID.

    Parameters
    ----------
    ids : list or numpy.ndarray
        List of IDs from which triads are generated. IDs can be integers or strings.
    baseid : int or str
        Base ID to which triads are pinned. Can be an integer or string.

    Returns
    -------
    list of tuple
        List of triads where each triad consists of three IDs.

    Raises
    ------
    TypeError
        If the input ids is not a list or numpy array.
        If the input baseid is not a scalar integer or string.
    ValueError
        If the input ids has fewer than 3 elements.
        If the input baseid is not found in the input ids.

    Notes
    -----
    This function generates triads of IDs by fixing one ID (baseid) and forming all possible combinations of two other IDs from the input list of IDs.

    Examples
    --------
    >>> generate_triangles([1, 2, 3, 4], 1)
    [(1, 2, 3), (1, 2, 4), (1, 3, 4)]
    >>> generate_triangles(['A', 'B', 'C', 'D'], 'A')
    [('A', 'B', 'C'), ('A', 'B', 'D'), ('A', 'C', 'D')]    
    """

    if not isinstance(ids, (list,NP.ndarray)):
        raise TypeError('Input ids must be a list or numpy array')
    ids = NP.asarray(ids).reshape(-1)
    if ids.size < 3:
        raise ValueError('Input ids must have at least  elements')
    if not isinstance(baseid, (int,str)):
        raise TypeError('Input baseid must be a scalar integer or string')
    if baseid not in ids:
        raise ValueError('Input baseid not found in inputs ids')
    ids = NP.unique(ids) # select all unique ids
    otherids = [id for id in ids if id!=baseid] # Find other ids except the baseid
    triads = [(baseid,oid1,oid2) for oid1_ind,oid1 in enumerate(otherids) for oid2_ind,oid2 in enumerate(otherids) if oid2_ind > oid1_ind]
    return triads

def unpack_realimag_to_real(inparr: NP.ndarray, arrtype: str = 'matrix') -> NP.ndarray:
    """
    Unpack the real and imaginary parts of a complex-valued numpy array into a real array.

    :param inparr: numpy.ndarray
        Input complex-valued array to be unpacked.
    :param arrtype: str, optional
        Type of output array. 'matrix' for 2x2 polarimetric case or 'scalar' for scalars.
        Default is 'matrix'.

    :return: numpy.ndarray
        Unpacked real array.

    :raises ValueError: If arrtype is neither 'matrix' nor 'scalar'.

    This function converts a complex-valued array into a real array by concatenating its real and imaginary parts along the last dimension.

    Examples
    --------
    >>> import numpy as np
    >>> arr = NP.array([1+2j, 3-4j])
    >>> unpack_realimag_to_real(arr)
    array([[1., 2.],
           [3., -4.]])

    When arrtype='matrix':
    >>> arr = NP.array([[[1+2j, 3+4j], [5+6j, 7+8j]], [[9+10j, 11+12j], [13+14j, 15+16j]]])
    >>> unpack_realimag_to_real(arr, arrtype='matrix')
    array([[ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.],
           [ 9., 10., 11., 12., 13., 14., 15., 16.]])
    """
    inparr = NP.asarray(inparr)
    inshape = inparr.shape
    print(inparr.reshape(-1)[0])
    outarr = NP.concatenate((inparr.real[..., NP.newaxis], inparr.imag[..., NP.newaxis]), axis=-1)

    if arrtype == 'matrix':
        outarr = outarr.reshape(inshape[:-3] + (-1,))
    elif arrtype == 'scalar':
        outarr = outarr.reshape(inshape[:-1] + (-1,))
    else:
        raise ValueError("Invalid arrtype. Should be 'matrix' or 'scalar'.")

    return outarr

def repack_real_to_realimag(inparr: NP.ndarray, arrtype: str = 'matrix') -> NP.ndarray:
    """
    Repack the unpacked real and imaginary parts into a complex-valued numpy array.

    :param inparr: numpy.ndarray
        Input real array to be repacked.
    :param arrtype: str, optional
        Type of input array. 'matrix' for 2x2 polarimetric case or 'scalar' for scalars.
        Default is 'matrix'.

    :return: numpy.ndarray
        Repacked complex-valued array.

    This function converts a real array into a complex-valued array by reshaping and combining its real and imaginary parts.
    If arrtype is 'matrix', the input array is assumed to represent 2x2 polarimetric case, otherwise, it represents scalars.

    Examples
    --------
    When arrtype='matrix':
    >>> import numpy as np
    >>> arr = NP.array([[1, 2], [3, -4]])
    >>> repack_real_to_realimag(arr, arrtype='matrix')
    array([[[ 1.+2.j,  3.+4.j],
            [ 5.+6.j,  7.+8.j]],
           [[ 9.+10.j, 11.+12.j],
            [13.+14.j, 15.+16.j]]])

    When arrtype='scalar':
    >>> arr = NP.array([[1, 2], [3, -4]])
    >>> repack_real_to_realimag(arr, arrtype='scalar')
    array([ 1.+2.j,  3.-4.j])
    """
    inparr = NP.asarray(inparr)
    inshape = inparr.shape
    outarr = inparr.reshape(inshape[:-1] + (-1, 2))  # Convert the array from (...,2N) to (...,N,2) real float values
    outarr = outarr[..., 0] + 1j * outarr[..., 1]  # (...,N) complex values
    if arrtype == 'matrix':
        outarr = outarr.reshape(inshape[:-1] + (-1, 2, 2))  # Convert from (...,4M) complex to (...,M,2,2) complex
    return outarr

def generate_independent_quads(ids: List[Union[int, str]]) -> List[List[Tuple[Union[int, str], Union[int, str]]]]:
    """
    Generate a list of independent quads from a list of IDs by forming pairs 
    of adjacent elements in a cyclic manner and concatenating pairs with 
    non-overlapping elements. 

    Algorithm based on L. Blackburn et al. (2020)

    Each quad is represented as a list of two 2-element tuples. Duplicate 
    quads with different orders, such as [(0, 1), (2, 3)] and [(2, 3), (0, 1)], 
    are removed.

    Parameters
    ----------
    ids : List[Union[int, str]]
        A list of IDs which can be integers or strings.

    Returns
    -------
    List[List[Tuple[Union[int, str], Union[int, str]]]]
        A list of independent quads, where each quad is a list of two 
        2-element tuples.
    
    Examples
    --------
    >>> ids = [0, 1, 2, 3, 4]
    >>> generate_quads(ids)
    [[(0, 1), (2, 3)], [(0, 1), (3, 4)], [(1, 2), (3, 4)], [(1, 2), (4, 0)], [(2, 3), (4, 0)]]
    
    >>> ids = ['a', 'b', 'c', 'd', 'e']
    >>> generate_quads(ids)
    [[('a', 'b'), ('c', 'd')], [('a', 'b'), ('d', 'e')], [('b', 'c'), ('d', 'e')], [('b', 'c'), ('e', 'a')], [('c', 'd'), ('e', 'a')]]
    """
    # Generate cyclic adjacent pairs
    pairs = [(ids[i], ids[(i + 1) % len(ids)]) for i in range(len(ids))]
    
    quads = []
    # Iterate through all combinations of pairs and check for independence
    for pair1, pair2 in itertools.combinations(pairs, 2):
        # Ensure pairs do not have overlapping elements
        if set(pair1).isdisjoint(pair2):
            # Sort the pairs to avoid duplicate quads in different orders
            quad = [tuple(pair1), tuple(pair2)]
            quads.append(sorted(quad))

    # Convert the set to a sorted list of lists
    return sorted(quads)

