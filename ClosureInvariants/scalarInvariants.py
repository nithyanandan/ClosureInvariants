import numpy as NP
from typing import List, Tuple, Union

def corrupt_visibilities(vis: NP.ndarray, g_a: NP.ndarray, g_b: NP.ndarray) -> NP.ndarray:
    """
    Corrupt visibilities with complex gains, g_a as a pre-factor and g_b as a post-factor.

    Parameters:
    vis : numpy.ndarray
        The input array representing visibilities of shape (...,Nbl).
    g_a : numpy.ndarray
        The complex antenna gains as a pre-factor of shape (...,Nbl).
    g_b : numpy.ndarray
        The complex antenna gains as a post-factor of shape (...,Nbl).

    Returns:
    numpy.ndarray
        The corrupted visibilities.

    Raises:
    TypeError: If vis, g_a, or g_b is not a numpy array.
    ValueError: If inputs have incompatible dimensions or shapes.

    Examples:
    >>> import numpy as NP
    >>> vis = NP.array([1+2j, 3+4j, 5+6j, 7+8j])
    >>> g_a = NP.array([0.9+0.1j, 1.0, 1.1, 0.95+0.05j])
    >>> g_b = np.array([1+0.2j, 1.0, 1.2, 0.9+0.1j])
    >>> corrupt_visibilities(vis, g_a, g_b)
    array([0.7 + 1.76j, 3 + 4j, 6.6 + 7.92j, 4.83 + 6.53j])
    """
    if not isinstance(vis, NP.ndarray):
        raise TypeError('Input vis must be a numpy array')
    if not isinstance(g_a, NP.ndarray):
        raise TypeError('Input g_a must be a numpy array')
    if not isinstance(g_b, NP.ndarray):
        raise TypeError('Input g_b must be a numpy array')
    if vis.ndim != g_a.ndim:
        raise ValueError('Inputs vis and g_a must have same number of dimensions')
    if vis.ndim != g_b.ndim:
        raise ValueError('Inputs vis and g_b must have same number of dimensions')
    if g_a.ndim != g_b.ndim:
        raise ValueError('Inputs g_a and g_b must have same number of dimensions')
    return g_a * vis * g_b.conj()

def corrs_list_on_loops(corrs: NP.ndarray, 
                        ant_pairs: List[Union[Tuple[Union[int, str], Union[int, str]], 
                                              List[Union[int, str]]]], 
                        loops: List[Union[List[Union[int, str]], Tuple[Union[int, str], ...]]], 
                        bl_axis: int = -1) -> List[List[NP.ndarray]]:
    """
    Generate triads of antenna correlations based on loops of antenna pairs.

    Parameters
    ----------
    corrs : numpy.ndarray
        Array of correlations associated with antenna pairs.
    ant_pairs : list of tuples or list of lists
        List of antenna pairs. Each pair is represented as a tuple or a list of two elements.
    loops : list of lists or tuple of lists/tuples
        List of loops of antenna indices. Each loop is represented as a list or a tuple of antenna indices.
    bl_axis : int, optional
        Axis corresponding to the number of antenna pairs in the correlations array. Default is -1.

    Returns
    -------
    list of lists of numpy.ndarray
        List of 3-element tuples of correlations, one tuple for each loop.

    Raises
    ------
    TypeError
        If the input ant_pairs or loops are not of valid types.
    ValueError
        If the input ant_pairs has invalid shape.
        If the input loops has invalid shape.
        If the input corrs and ant_pairs do not have the same number of baselines.

    Notes
    -----
    This function generates sets of antenna correlations by picking three sets of antenna pairs from the input list of antenna pairs, 
    based on the specified loops of antenna indices.

    Examples
    --------
    >>> corrs = np.random.randn(10)  # Example correlations array
    >>> ant_pairs = [(0, 1), (1, 2), (2, 3), (3, 4)]  # Example antenna pairs
    >>> loops = [[0, 1, 2], [2, 3, 4]]  # Example loops of antenna indices
    >>> corrs_list_on_loops(corrs, ant_pairs, loops)
    [[[correlation1], [correlation2], [correlation3]],
     [[correlation4], [correlation5], [correlation6]]]
    """

    if not isinstance(ant_pairs, (list,NP.ndarray)):
        raise TypeError('Input ant_pairs must be a list or numpy array')
    ant_pairs = NP.asarray(ant_pairs)
    if ant_pairs.ndim == 1:
        if ant_pairs.size != 2:
            raise ValueError('Input ant_pairs contains invalid shape')
    elif ant_pairs.ndim == 2:
        if ant_pairs.shape[-1] != 2:
            raise ValueError('Input ant_pairs contains invalid shape')
    else:
        raise ValueError('Input ant_pairs contains invalid shape')
    ant_pairs = NP.asarray(ant_pairs).reshape(-1,2)
    
    if not isinstance(bl_axis, int):
        raise TypeError('Input bl_axis must be an integer')
        
    if not isinstance(corrs, NP.ndarray):
        raise TypeError('Input corrs must be a numpy array')
    if corrs.shape[bl_axis] != ant_pairs.shape[0]:
        raise ValueError('Input corrs and ant_pairs do not have same number of baselines')
    
    if not isinstance(loops, (list,NP.ndarray)):
        raise TypeError('Input loops must be a list or numpy array')
    loops = NP.asarray(loops)
    if loops.ndim == 1:
        loops = loops.reshape(1,-1)
    elif loops.ndim != 2:
        raise ValueError('Input loops contains invalid shape')

    corrs_lol = []
    for loopi, loop in enumerate(loops):
        corrs_loop = []
        for i in range(len(loop)):
            bl_ind = NP.where((ant_pairs[:,0] == loop[i]) & (ant_pairs[:,1]==loop[(i+1)%loop.size]))[0]
            if bl_ind.size == 1:
                corr = NP.copy(NP.take(corrs, bl_ind, axis=bl_axis))
            elif bl_ind.size == 0: # Check for reversed pair
                bl_ind = NP.where((ant_pairs[:,0] == loop[(i+1)%loop.size]) & (ant_pairs[:,1]==loop[i]))[0]
                if bl_ind.size == 0:
                    raise IndexError('Specified antenna pair ({0:0d},{1:0d}) not found in input ant_pairs'.format(loop[i], loop[(i+1)%loop.size]))
                elif bl_ind.size == 1: # Take conjugate
                    corr = NP.take(corrs.conj(), bl_ind, axis=bl_axis)
                elif bl_ind.size > 1:
                    raise IndexError('{0:0d} indices found for antenna pair ({1:0d},{2:0d}) in input ant_pairs'.format(bl_ind, loop[i], loop[(i+1)%loop.size]))
            elif bl_ind.size > 1:
                raise IndexError('{0:0d} indices found for antenna pair ({1:0d},{2:0d}) in input ant_pairs'.format(bl_ind, loop[i], loop[(i+1)%loop.size]))
        
            corr = NP.take(corr, 0, axis=bl_axis)
            corrs_loop += [corr]
        corrs_lol += [corrs_loop]
        
    return corrs_lol

def advariant(corrs_list: Union[List[List[NP.ndarray]], List[NP.ndarray]]) -> NP.ndarray:
    """
    Construct the advariant from a list of odd-numbered correlations forming a closed odd-edged loop.

    Parameters
    ----------
    corrs_list : List or numpy.ndarray of numpy.ndarray
        List of odd-numbered correlations forming the edges of a closed odd-edged loop.

    Returns
    -------
    numpy.ndarray
        Advariant constructed from the list of correlations.

    Raises
    ------
    TypeError
        If the input corrs_list is not a list.
        If any element of corrs_list is not a numpy array.
    ValueError
        If the input corrs_list does not have an odd number of elements.
        If the shapes of numpy arrays in corrs_list are not identical.

    Notes
    -----
    The advariant is constructed by taking the product of correlations from an odd-numbered list of 
    correlations that forms the edges of a closed odd-edged loop. Every alternate correlation, starting from 
    the second element, undergoes the hat operation, which involves taking the inverse conjugate.

    Examples
    --------
    >>> import numpy as np
    >>> from your_module import advariant
    >>> corrs_list = np.array([5j, 3+2j, 1-4j])
    >>> advariant(corrs_list)
    array([3.84615385+4.23076923j])
    """

    if not isinstance(corrs_list, (list,NP.ndarray)):
        raise TypeError('Input corrs_list must be a list')
    nedges = len(corrs_list)
    if nedges%2 == 0:
        raise ValueError('Input corrs_list must be a list made of odd number of elements for an advariant to be constructed')

    advar = None
    for edgei,corr in enumerate(corrs_list):  
        if edgei == 0:
            advar = NP.copy(corr)
        else:
            if edgei%2 == 0:
                advar = advar * corr
            else:
                advar = advar / corr.conj()
                
    return advar

def advariants_multiple_loops(corrs_lol: List[List[NP.ndarray]]) -> NP.ndarray:
    """
    Calculate advariants on multiple loops from a list of lists of odd-numbered correlations forming closed odd-edged loops.

    Parameters
    ----------
    corrs_lol : List of lists of numpy.ndarray
        List of lists of odd-numbered correlations forming closed odd-edged loops.

    Returns
    -------
    numpy.ndarray
        Advariants calculated for each loop in the input list.

    Raises
    ------
    TypeError
        If the input corrs_lol is not a list of lists.
    ValueError
        If the input corrs_lol contains invalid shapes of numpy arrays.
        If the input corrs_lol is empty.

    Notes
    -----
    This function calculates advariants on multiple loops by calling the advariant function for each loop in the input 
    list of lists of odd-numbered correlations. The hat operation, which involves taking the inverse of the conjugate, 
    is applied during the advariant calculation.

    Examples
    --------
    >>> import numpy as np
    >>> from your_module import advariants_multiple_loops
    >>> cl1 = [NP.array([1+2j, 3+4j, 5+6j]),
    ...        NP.array([7+8j, 9+10j, 11+12j]),
    ...        NP.array([13+14j, 15+16j, 17+18j])]
    >>> cl2 = [NP.array([25+26j, 27+28j, 29+30j]),
    ...        NP.array([31+32j, 33+34j, 35+36j]),
    ...        NP.array([37+38j, 39+40j, 41+42j])]
    >>> cl3 = [NP.array([49+50j, 51+52j, 53+54j]),
    ...        NP.array([55+56j, 57+58j, 59+60j]),
    ...        NP.array([61+62j, 63+64j, 65+66j])]
    >>> corrs_lol = [cl1, cl2, cl3]
    >>> advariants_multiple_loops(corrs_lol)
    array([[ -3.76106195 +1.4159292j   -6.91160221 +4.32044199j
             -9.6490566  +6.92830189j]
           [-31.8070529 +28.84433249j -33.87928731+30.91224944j
            -35.94327648+32.97262991j]
           [-56.32738192+53.33939296j -58.35097535+55.36216543j
            -60.37296992+57.38342042j]])
    """
    if not isinstance(corrs_lol, list):
        raise TypeError('Input corrs_lol must be a list of lists')
    advars_list = []
    for ind, corrs_list in enumerate(corrs_lol):
        advars_list += [advariant(corrs_list)]
    return NP.array(advars_list)
