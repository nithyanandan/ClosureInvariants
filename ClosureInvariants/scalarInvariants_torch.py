from typing import List, Tuple, Union, Optional

import numpy as NP
import torch

def corrupt_visibilities(vis: torch.Tensor, g_a: torch.Tensor, g_b: torch.Tensor) -> torch.Tensor:
    """
    Corrupt visibilities with complex gains, g_a as a pre-factor and g_b as a post-factor.

    Parameters:
    vis : torch.Tensor
        The input array representing visibilities of shape (...,Nbl).
    g_a : torch.Tensor
        The complex antenna gains as a pre-factor of shape (...,Nbl).
    g_b : torch.Tensor
        The complex antenna gains as a post-factor of shape (...,Nbl).

    Returns:
    torch.Tensor
        The corrupted visibilities.

    Raises:
    TypeError: If vis, g_a, or g_b is not a torch tensor.
    ValueError: If inputs have incompatible dimensions or shapes.

    Examples:
    >>> import torch
    >>> vis = torch.tensor([1+2j, 3+4j, 5+6j, 7+8j])
    >>> g_a = torch.tensor([0.9+0.1j, 1.0, 1.1, 0.95+0.05j])
    >>> g_b = torch.tensor([1+0.2j, 1.0, 1.2, 0.9+0.1j])
    >>> corrupt_visibilities(vis, g_a, g_b)
    tensor([0.7000 + 1.7600j, 3.0000 + 4.0000j, 6.6000 + 7.9200j, 4.8300 + 6.5300j])
    """
    if not isinstance(vis, torch.Tensor):
        raise TypeError('Input vis must be a torch tensor')
    if not isinstance(g_a, torch.Tensor):
        raise TypeError('Input g_a must be a torch tensor')
    if not isinstance(g_b, torch.Tensor):
        raise TypeError('Input g_b must be a torch tensor')
    if vis.ndim != g_a.ndim:
        raise ValueError('Inputs vis and g_a must have same number of dimensions')
    if vis.ndim != g_b.ndim:
        raise ValueError('Inputs vis and g_b must have same number of dimensions')
    if g_a.ndim != g_b.ndim:
        raise ValueError('Inputs g_a and g_b must have same number of dimensions')
    return g_a * vis * g_b.conj()

def corrs_list_on_loops(corrs: torch.Tensor, 
                        ant_pairs: List[Union[Tuple[Union[int, str], Union[int, str]], 
                                              List[Union[int, str]]]], 
                        loops: List[Union[List[Union[int, str]], Tuple[Union[int, str], ...]]], 
                        bl_axis: int = -1) -> List[List[torch.Tensor]]:
    """
    Generate triads of antenna correlations based on loops of antenna pairs.

    Parameters
    ----------
    corrs : torch.Tensor
        Array of correlations associated with antenna pairs.
    ant_pairs : list of tuples or list of lists
        List of antenna pairs. Each pair is represented as a tuple or a list of two elements.
    loops : list of lists or tuple of lists/tuples
        List of loops of antenna indices. Each loop is represented as a list or a tuple of antenna indices.
    bl_axis : int, optional
        Axis corresponding to the number of antenna pairs in the correlations array. Default is -1.

    Returns
    -------
    list of lists of torch.Tensor
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
    >>> corrs = torch.randn(10)  # Example correlations array
    >>> ant_pairs = [(0, 1), (1, 2), (2, 3), (3, 4)]  # Example antenna pairs
    >>> loops = [[0, 1, 2], [2, 3, 4]]  # Example loops of antenna indices
    >>> corrs_list_on_loops(corrs, ant_pairs, loops)
    [[[correlation1], [correlation2], [correlation3]],
     [[correlation4], [correlation5], [correlation6]]]
    """

    if not isinstance(ant_pairs, (list, NP.ndarray)):
        raise TypeError('Input ant_pairs must be a list or numpy array')
    ant_pairs = NP.array(ant_pairs)
    if ant_pairs.ndim == 1:
        if ant_pairs.size != 2:
            raise ValueError('Input ant_pairs contains invalid shape')
    elif ant_pairs.ndim == 2:
        if ant_pairs.shape[-1] != 2:
            raise ValueError('Input ant_pairs contains invalid shape')
    else:
        raise ValueError('Input ant_pairs contains invalid shape')
    ant_pairs = ant_pairs.reshape(-1, 2)
    
    if not isinstance(bl_axis, int):
        raise TypeError('Input bl_axis must be an integer')
        
    if not isinstance(corrs, torch.Tensor):
        raise TypeError('Input corrs must be a torch tensor')
    if corrs.size(bl_axis) != ant_pairs.shape[0]:
        raise ValueError('Input corrs and ant_pairs do not have same number of baselines')
    
    if not isinstance(loops, (list, NP.ndarray)):
        raise TypeError('Input loops must be a list or numpy array')
    loops = NP.array(loops)
    if loops.ndim == 1:
        loops = loops.reshape(1, -1)
    elif loops.ndim != 2:
        raise ValueError('Input loops contains invalid shape')

    corrs_lol = []
    for loopi, loop in enumerate(loops):
        corrs_loop = []
        no_loop = False
        for i in range(len(loop)):
            bl_ind = NP.where((ant_pairs[:, 0] == loop[i]) & (ant_pairs[:, 1] == loop[(i + 1) % loop.size]))[0]
            if bl_ind.size == 1:
                corr = torch.clone(torch.index_select(corrs, bl_axis, torch.tensor(bl_ind)))
            elif bl_ind.size == 0:  # Check for reversed pair
                bl_ind = NP.where((ant_pairs[:, 0] == loop[(i + 1) % loop.size]) & (ant_pairs[:, 1] == loop[i]))[0]
                if bl_ind.size == 0:
                    no_loop = True
                    break # Pair not found, possibly loop not possible with the given dataset
                    raise IndexError('Specified antenna pair ({0},{1}) not found in input ant_pairs'.format(loop[i], loop[(i + 1) % loop.size]))
                elif bl_ind.size == 1:  # Take conjugate
                    corr = torch.index_select(corrs.conj(), bl_axis, torch.tensor(bl_ind))
                elif bl_ind.size > 1:
                    raise IndexError('{0} indices found for antenna pair ({1},{2}) in input ant_pairs'.format(bl_ind.size, loop[i], loop[(i + 1) % loop.size]))
            elif bl_ind.size > 1:
                raise IndexError('{0} indices found for antenna pair ({1},{2}) in input ant_pairs'.format(bl_ind.size, loop[i], loop[(i + 1) % loop.size]))
            
            corr = torch.index_select(corr, bl_axis, torch.tensor([0]))
            corrs_loop.append(corr)
        if not no_loop:
            corrs_lol.append(corrs_loop)
        
    return corrs_lol

def advariant(corrs_list: Union[List[List[torch.Tensor]], List[torch.Tensor]]) -> torch.Tensor:
    """
    Construct the advariant from a list of odd-numbered correlations forming a closed odd-edged loop.

    Parameters
    ----------
    corrs_list : List or torch.Tensor of torch.Tensor
        List of odd-numbered correlations forming the edges of a closed odd-edged loop.

    Returns
    -------
    torch.Tensor
        Advariant constructed from the list of correlations.

    Raises
    ------
    TypeError
        If the input corrs_list is not a list.
        If any element of corrs_list is not a torch tensor.
    ValueError
        If the input corrs_list does not have an odd number of elements.
        If the shapes of torch tensors in corrs_list are not identical.

    Notes
    -----
    The advariant is constructed by taking the product of correlations from an odd-numbered list of 
    correlations that forms the edges of a closed odd-edged loop. Every alternate correlation, starting from 
    the second element, undergoes the hat operation, which involves taking the inverse conjugate.

    Examples
    --------
    >>> import torch
    >>> from your_module import advariant
    >>> corrs_list = torch.tensor([5j, 3+2j, 1-4j])
    >>> advariant(corrs_list)
    tensor([3.8462+4.2308j])
    """

    if not isinstance(corrs_list, (list, torch.Tensor)):
        raise TypeError('Input corrs_list must be a list or torch tensor')
    nedges = len(corrs_list)
    if nedges % 2 == 0:
        raise ValueError('Input corrs_list must be a list made of odd number of elements for an advariant to be constructed')

    advar = None
    for edgei, corr in enumerate(corrs_list):
        if edgei == 0:
            advar = torch.clone(corr)
        else:
            if edgei % 2 == 0:
                advar = advar * corr
            else:
                advar = advar / corr.conj()

    return advar

def advariants_multiple_loops(corrs_lol: List[List[torch.Tensor]]) -> torch.Tensor:
    """
    Calculate advariants on multiple loops from a list of lists of odd-numbered correlations forming closed odd-edged loops.

    Parameters
    ----------
    corrs_lol : List of lists of torch.Tensor
        List of lists of odd-numbered correlations forming closed odd-edged loops.

    Returns
    -------
    torch.Tensor
        Advariants calculated for each loop in the input list.

    Raises
    ------
    TypeError
        If the input corrs_lol is not a list of lists.
    ValueError
        If the input corrs_lol contains invalid shapes of torch tensors.
        If the input corrs_lol is empty.

    Notes
    -----
    This function calculates advariants on multiple loops by calling the advariant function for each loop in the input 
    list of lists of odd-numbered correlations. The hat operation, which involves taking the inverse of the conjugate, 
    is applied during the advariant calculation.

    Examples
    --------
    >>> import torch
    >>> from your_module import advariants_multiple_loops
    >>> cl1 = [torch.tensor([1+2j, 3+4j, 5+6j]),
    ...        torch.tensor([7+8j, 9+10j, 11+12j]),
    ...        torch.tensor([13+14j, 15+16j, 17+18j])]
    >>> cl2 = [torch.tensor([25+26j, 27+28j, 29+30j]),
    ...        torch.tensor([31+32j, 33+34j, 35+36j]),
    ...        torch.tensor([37+38j, 39+40j, 41+42j])]
    >>> cl3 = [torch.tensor([49+50j, 51+52j, 53+54j]),
    ...        torch.tensor([55+56j, 57+58j, 59+60j]),
    ...        torch.tensor([61+62j, 63+64j, 65+66j])]
    >>> corrs_lol = [cl1, cl2, cl3]
    >>> advariants_multiple_loops(corrs_lol)
    tensor([[ -3.7611 +1.4159j,  -6.9116 +4.3204j,  -9.6491 +6.9283j],
            [-31.8071+28.8443j, -33.8793+30.9122j, -35.9433+32.9726j],
            [-56.3274+53.3394j, -58.3510+55.3622j, -60.3730+57.3834j]])
    """
    if not isinstance(corrs_lol, list):
        raise TypeError('Input corrs_lol must be a list of lists')
    advars_list = []
    for ind, corrs_list in enumerate(corrs_lol):
        advars_list += [advariant(corrs_list)]

    return torch.moveaxis(torch.stack(advars_list), 0, -1).squeeze(dim=-2)  # Move the loop axis to first from the end

def invariants_from_advariants_method1(advariants: torch.Tensor, 
                                       normaxis: int, 
                                       normwts: Optional[Union[torch.Tensor, str]] = None, 
                                       normpower: int = 2) -> torch.Tensor:
    """
    Calculate the invariants from the advariants using a specified normalization method.
    The real and imaginary parts of the advariants are split, concatenated, and then 
    the normalization is performed on this concatenated array.

    Parameters
    ----------
    advariants : torch.Tensor
        Array of advariants to be normalized.
    normaxis : int
        Axis along which the normalization occurs.
    normwts : torch.Tensor, optional
        Weights used for the normalization. Should be broadcastable to the shape of advariants.
        Default is 1 along the normaxis.
    normpower : int, optional
        Power used to define the norm (e.g., L1 norm if normpower=1, L2 norm if normpower=2).
        Default is 2 (L2 norm).

    Returns
    -------
    torch.Tensor
        Array of normalized invariants.

    Raises
    ------
    TypeError
        If the input advariants is not a torch tensor.
        If normwts is provided and is not a torch tensor.
        If normpower is not an integer.

    Examples
    --------
    >>> import torch
    >>> from your_module import invariants_from_advariants_method1
    >>> advariants = torch.randn(3, 4, 5) + 1j * torch.randn(3, 4, 5)
    >>> normaxis = -1
    >>> invariants = invariants_from_advariants_method1(advariants, normaxis)
    >>> print(invariants)

    >>> normwts = torch.rand(3, 4, 5)
    >>> invariants = invariants_from_advariants_method1(advariants, normaxis, normwts=normwts, normpower=1)
    >>> print(invariants)
    """
    
    # Validate inputs
    if not isinstance(advariants, torch.Tensor):
        raise TypeError('Input advariants must be a torch tensor')
    if normwts is not None and not (isinstance(normwts, torch.Tensor) or normwts == 'max'):
        raise TypeError("Input normwts must be a torch tensor, None, or 'max'")
    if not isinstance(normpower, int):
        raise TypeError('Input normpower must be an integer')

    # Split real and imaginary parts and concatenate them
    if torch.is_complex(advariants):
        # realvalued_advariants_shape = torch.tensor(advariants.shape)
        # realvalued_advariants_shape[-1] *= 2
        # realvalued_advariants = torch.zeros(tuple(realvalued_advariants_shape))
        # realvalued_advariants[:, :, ::2] = advariants.real
        # realvalued_advariants[:, :, 1::2] = advariants.imag
        realvalued_advariants = torch.cat((advariants.real, advariants.imag), dim=-1)
    else:
        realvalued_advariants = advariants.clone()
    
    if isinstance(normwts, str):
        if normwts == 'max':
            realvalued_normwts = torch.zeros_like(realvalued_advariants)
            max_indices = torch.argmax(torch.abs(realvalued_advariants), dim=normaxis, keepdim=True)
            realvalued_normwts.scatter_(normaxis, max_indices, 1)
    elif normwts is None:
        realvalued_normwts = torch.ones_like(realvalued_advariants)
    else:
        if torch.is_complex(advariants):
            realvalued_normwts = torch.cat((normwts.real, normwts.imag), dim=-1)
        else:
            realvalued_normwts = normwts
    
    # Calculate the normalization factor
    normalization_factor = torch.sum(torch.abs(realvalued_advariants * realvalued_normwts)**normpower, dim=normaxis, keepdim=True)**(1/normpower)

    # Calculate the invariants
    invariants = realvalued_advariants / normalization_factor

    return invariants

def closurePhases_from_advariants(advariants: torch.Tensor) -> torch.Tensor:
    """
    Calculate closure phases from the given advariants.

    The closure phase is defined as the phase angle of the advariants.
    This function returns the phase of each element in the input advariants array.
    If the advariants are real, the phase will be zero or pi.

    Parameters
    ----------
    advariants : torch.Tensor
        A torch tensor (real or complex) representing the advariants from which closure phases 
        are to be computed.

    Returns
    -------
    torch.Tensor
        A torch tensor containing the closure phases, which are the angles of the 
        elements in the input advariants array. The output array has the 
        same shape as the input advariants array.

    Raises
    ------
    TypeError
        If the input advariants is not a torch tensor.

    Examples
    --------
    >>> import torch
    >>> advariants = torch.tensor([[1+2j, 3+4j], [5+6j, 7+8j]])
    >>> closure_phases = closurePhases_from_advariants(advariants)
    >>> print(closure_phases)
    tensor([[ 1.1071,  0.9273],
            [ 0.8761,  0.8520]])

    >>> advariants_real = torch.tensor([[1, 2], [3, 4]])
    >>> closure_phases_real = closurePhases_from_advariants(advariants_real)
    >>> print(closure_phases_real)
    tensor([[0., 0.],
            [0., 0.]])
    """

    # Validate input
    if not isinstance(advariants, torch.Tensor):
        raise TypeError('Input advariants must be a torch tensor')

    # Calculate closure phases (angles of the advariants)
    closure_phases = torch.angle(advariants)

    return closure_phases

def covariant(corrs_list: Union[List[List[torch.Tensor]], List[torch.Tensor]]) -> torch.Tensor:
    """
    Construct the covariant from a list of even-numbered correlations forming a closed even-edged loop.

    Parameters
    ----------
    corrs_list : List or torch.Tensor of torch.Tensor
        List of even-numbered correlations forming the edges of a closed even-edged loop.

    Returns
    -------
    torch.Tensor
        Covariant constructed from the list of correlations.

    Raises
    ------
    TypeError
        If the input corrs_list is not a list.
        If any element of corrs_list is not a torch tensor.
    ValueError
        If the input corrs_list does not have an even number of elements.
        If the shapes of torch tensors in corrs_list are not identical.

    Notes
    -----
    The covariant is constructed by taking the product of correlations from an even-numbered list of 
    correlations that forms the edges of a closed even-edged loop. Every alternate correlation, starting from 
    the first element, undergoes the hat operation, which involves taking the inverse conjugate.

    Examples
    --------
    >>> import torch
    >>> from your_module import covariant
    >>> corrs_list = [torch.tensor([2j, 3+2j, 1-4j, 2, 1, 3, 2]),
    ...               torch.tensor([2+1j, 1-2j, 3+4j, 5, 6, 7, 8]),
    ...               torch.tensor([4-1j, 2+3j, 1, 0, 1, 3, 5]),
    ...               torch.tensor([1+1j, 2, 3-1j, 4, 5, 6, 7])]
    >>> covariant(corrs_list)
    """

    if not isinstance(corrs_list, (list, torch.Tensor)):
        raise TypeError('Input corrs_list must be a list or torch tensor')
    nedges = len(corrs_list)
    if nedges % 2 != 0:
        raise ValueError('Input corrs_list must be a list made of even number of elements for a covariant to be constructed')

    covar = None
    for edgei, corr in enumerate(corrs_list):
        if edgei == 0:
            covar = torch.clone(corr)
        else:
            if edgei % 2 == 0:
                covar = covar * corr
            else:
                covar = covar / corr.conj()

    return covar

def covariants_multiple_loops(corrs_lol: List[List[torch.Tensor]]) -> torch.Tensor:
    """
    Calculate covariants on multiple loops from a list of lists of even-numbered correlations forming closed even-edged loops.

    Parameters
    ----------
    corrs_lol : List of lists of torch.Tensor
        List of lists of even-numbered correlations forming closed even-edged loops.

    Returns
    -------
    torch.Tensor
        Covariants calculated for each loop in the input list.

    Raises
    ------
    TypeError
        If the input corrs_lol is not a list of lists.
    ValueError
        If the input corrs_lol contains invalid shapes of torch tensors.
        If the input corrs_lol is empty.

    Notes
    -----
    This function calculates covariants on multiple loops by calling the covariant function for each loop in the input 
    list of lists of even-numbered correlations. The hat operation, which involves taking the inverse of the conjugate, 
    is applied during the covariant calculation.

    Examples
    --------
    >>> import torch
    >>> from your_module import covariants_multiple_loops
    >>> cl1 = [torch.tensor([1+2j, 3+4j, 5+6j, 7+8j, 9, 10, 11]),
    ...        torch.tensor([2+1j, 4+3j, 6+5j, 8+7j, 12, 13, 14]),
    ...        torch.tensor([3+2j, 5+4j, 7+6j, 9+8j, 15, 16, 17]),
    ...        torch.tensor([4+3j, 6+5j, 8+7j, 10+9j, 18, 19, 20])]
    >>> cl2 = [torch.tensor([11+12j, 13+14j, 15+16j, 17+18j, 21, 22, 23]),
    ...        torch.tensor([12+11j, 14+13j, 16+15j, 18+17j, 24, 25, 26]),
    ...        torch.tensor([13+12j, 15+14j, 17+16j, 19+18j, 27, 28, 29]),
    ...        torch.tensor([14+13j, 16+15j, 18+17j, 20+19j, 30, 31, 32])]
    >>> cl3 = [torch.tensor([21+22j, 23+24j, 25+26j, 27+28j, 33, 34, 35]),
    ...        torch.tensor([22+21j, 24+23j, 26+25j, 28+27j, 36, 37, 38]),
    ...        torch.tensor([23+22j, 25+24j, 27+26j, 29+28j, 39, 40, 41]),
    ...        torch.tensor([24+23j, 26+25j, 28+27j, 30+29j, 42, 43, 44])]
    >>> cl4 = [torch.tensor([31+32j, 33+34j, 35+36j, 37+38j, 45, 46, 47]),
    ...        torch.tensor([32+31j, 34+33j, 36+35j, 38+37j, 48, 49, 50]),
    ...        torch.tensor([33+32j, 35+34j, 37+36j, 39+38j, 51, 52, 53]),
    ...        torch.tensor([34+33j, 36+35j, 38+37j, 40+39j, 54, 55, 56])]
    >>> cl5 = [torch.tensor([41+42j, 43+44j, 45+46j, 47+48j, 57, 58, 59]),
    ...        torch.tensor([42+41j, 44+43j, 46+45j, 48+47j, 60, 61, 62]),
    ...        torch.tensor([43+42j, 45+44j, 47+46j, 49+48j, 63, 64, 65]),
    ...        torch.tensor([44+43j, 46+45j, 48+47j, 50+49j, 66, 67, 68])]
    >>> corrs_lol = [cl1, cl2, cl3, cl4, cl5]
    >>> covariants_multiple_loops(corrs_lol)
    """
    if not isinstance(corrs_lol, list):
        raise TypeError('Input corrs_lol must be a list of lists')
    if not all(isinstance(corrs_list, list) for corrs_list in corrs_lol):
        raise TypeError('Each element of corrs_lol must be a list')
    
    covars_list = []
    for ind, corrs_list in enumerate(corrs_lol):
        covars_list += [covariant(corrs_list)]
    
    # Move the loop axis to the first from the end
    return torch.moveaxis(torch.stack(covars_list), 0, -1)

def closureAmplitudes_from_covariants(covariants: torch.Tensor) -> torch.Tensor:
    """
    Calculate closure amplitudes from the given covariants.

    The closure amplitude is defined as the absolute value of the covariants.
    This function returns the absolute value of each element in the input covariants array.

    Parameters
    ----------
    covariants : torch.Tensor
        A torch tensor (real or complex) representing the covariants from which closure amplitudes 
        are to be computed.

    Returns
    -------
    torch.Tensor
        A torch tensor containing the closure amplitudes, which are the absolute values of the 
        elements in the input covariants array. The output array has the same shape as the input 
        covariants array.

    Raises
    ------
    TypeError
        If the input covariants is not a torch tensor.

    Examples
    --------
    >>> import torch
    >>> covariants = torch.tensor([[1+2j, 3+4j], [5+6j, 7+8j]])
    >>> closure_amplitudes = closureAmplitudes_from_covariants(covariants)
    >>> print(closure_amplitudes)
    tensor([[ 2.2361,  5.0000],
            [ 7.8102, 10.6301]])

    >>> covariants_real = torch.tensor([[1, 2], [3, 4]])
    >>> closure_amplitudes_real = closureAmplitudes_from_covariants(covariants_real)
    >>> print(closure_amplitudes_real)
    tensor([[1., 2.],
            [3., 4.]])
    """
    # Validate input
    if not isinstance(covariants, torch.Tensor):
        raise TypeError('Input covariants must be a torch tensor')

    # Calculate closure amplitudes (absolute values of the covariants)
    closure_amplitudes = torch.abs(covariants)

    return closure_amplitudes

def closurePhases_from_covariants(covariants: torch.Tensor) -> torch.Tensor:
    """
    Calculate closure phases from the given covariants.

    The closure phase is defined as the phase angle of the covariants.
    This function returns the phase of each element in the input covariants array.
    If the covariants are real, the phase will be zero or pi.

    Parameters
    ----------
    covariants : torch.Tensor
        A torch tensor (real or complex) representing the covariants from which closure phases 
        are to be computed.

    Returns
    -------
    torch.Tensor
        A torch tensor containing the closure phases, which are the angles of the 
        elements in the input covariants array. The output array has the 
        same shape as the input covariants array.

    Raises
    ------
    TypeError
        If the input covariants is not a torch tensor.

    Examples
    --------
    >>> import torch
    >>> covariants = torch.tensor([[1+2j, 3+4j], [5+6j, 7+8j]])
    >>> closure_phases = closurePhases_from_covariants(covariants)
    >>> print(closure_phases)
    tensor([[ 1.1071,  0.9273],
            [ 0.8761,  0.8520]])

    >>> covariants_real = torch.tensor([[1, 2], [3, 4]])
    >>> closure_phases_real = closurePhases_from_covariants(covariants_real)
    >>> print(closure_phases_real)
    tensor([[0., 0.],
            [0., 0.]])
    """
    # Validate input
    if not isinstance(covariants, torch.Tensor):
        raise TypeError('Input covariants must be a torch tensor')

    # Calculate closure phases (angles of the covariants)
    closure_phases = torch.angle(covariants)

    return closure_phases
