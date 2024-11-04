import numpy as NP
from typing import List, Tuple, Union, Optional
import torch

def corrupt_visibilities(vis: torch.Tensor, g_a: torch.Tensor, g_b: torch.Tensor, pol_axes: torch.Tensor = None) -> torch.Tensor:
    """
    Corrupt visibilities with complex gains, g_a as a pre-factor and g_b as a post-factor.
    The matrix multiplication occurs on the pol_axes specified.

    Parameters:
    vis : torch.Tensor
        The input array representing visibilities of shape (...,Nbl,2,2).
    g_a : torch.Tensor
        The complex antenna gains as a pre-factor of shape (...,Nbl,2,2).
    g_b : torch.Tensor
        The complex antenna gains as a post-factor of shape (...,Nbl,2,2).
    pol_axes : torch.Tensor, optional
        The axes over which the Hermitian conjugate is applied. 
        If None, Hermitian conjugate is applied over the last two axes.

    Returns:
    torch.Tensor
        The corrupted visibilities.

    Raises:
    TypeError: If vis, g_a, or g_b is not a torch tensor.
    ValueError: If inputs have incompatible dimensions or shapes.

    Examples:
    >>> import torch
    >>> vis = torch.tensor([[[1+2j, 3+4j], [5+6j, 7+8j]]])
    >>> g_a = torch.tensor([[[2+1j, 4-3j], [1+2j, 3+4j]]])
    >>> g_b = torch.tensor([[[3-2j, 5+4j], [2+3j, 4-5j]]])
    >>> corrupt_visibilities(vis, g_a, g_b)
    tensor([[[ 444. +12.j,  224.+272.j],
             [  48.+476.j, -272.+288.j]]])
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
    if vis.shape[-2:] != (2,2):
        raise ValueError('The last two axes of vis must have shape (2,2)')
    if g_a.shape[-2:] != (2,2):
        raise ValueError('The last two axes of g_a must have shape (2,2)')
    if g_b.shape[-2:] != (2,2):
        raise ValueError('The last two axes of g_b must have shape (2,2)')
    return g_a @ vis @ hermitian(g_b, axes=pol_axes)


def corrs_list_on_loops(corrs: torch.Tensor, 
                        ant_pairs: List[Union[Tuple[Union[int, str], Union[int, str]], 
                                              List[Union[int, str]]]], 
                        loops: List[Union[List[Union[int, str]], Tuple[Union[int, str], ...]]], 
                        bl_axis: int = -3, 
                        pol_axes: Union[List[int], Tuple[int, int]] = (-2, -1)) -> List[List[torch.Tensor]]:
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
        Axis corresponding to the number of antenna pairs in the correlations array. Default is -3.
    pol_axes : list or tuple of int, optional
        Axes corresponding to the polarization axes in the correlations array. Default is (-2, -1).

    Returns
    -------
    list of lists of torch.Tensor
        List of 3-element tuples of correlations, one tuple for each loop.

    Raises
    ------
    TypeError
        If the input ant_pairs or loops are not of valid types.
        If the input pol_axes is not a list, tuple, or torch tensor.
    ValueError
        If the input ant_pairs has invalid shape.
        If the input loops has invalid shape.
        If the input corrs and ant_pairs do not have the same number of baselines.
        If the input pol_axes does not have exactly two elements.

    Notes
    -----
    This function generates sets of antenna correlations by picking three sets of antenna pairs from the input list of antenna pairs, 
    based on the specified loops of antenna indices.

    Examples
    --------
    >>> corrs = torch.randn(10, 2, 2)  # Example correlations array
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
    if corrs.shape[bl_axis] != ant_pairs.shape[0]:
        raise ValueError('Input corrs and ant_pairs do not have same number of baselines')
    
    if not isinstance(loops, (list, NP.ndarray)):
        raise TypeError('Input loops must be a list or numpy array')
    loops = NP.array(loops)
    if loops.ndim == 1:
        loops = loops.reshape(1, -1)
    elif loops.ndim != 2:
        raise ValueError('Input loops contains invalid shape')

    if not isinstance(pol_axes, (list, tuple, torch.Tensor)):
        raise TypeError('Input pol_axes must be a list, tuple, or torch tensor')
    if len(pol_axes) != 2:
        raise ValueError('Input pol_axes must be a two-element sequence')
    else:
        for pax in pol_axes:
            if not isinstance(pax, int):
                raise TypeError('Input pol_axes must be a two-element sequence of integers')
    pol_axes = torch.tensor(pol_axes).ravel()
    tmpind = torch.where(pol_axes < 0)[0]
    if tmpind.size().numel() > 0:
        pol_axes[tmpind] += corrs.ndim # Convert to a positive value for the polarization axes
        
    corrs_lol = []
    for loopi, loop in enumerate(loops):
        corrs_loop = []
        for i in range(len(loop)):
            bl_ind = NP.where((ant_pairs[:, 0] == loop[i]) & (ant_pairs[:, 1] == loop[(i + 1) % loop.size]))[0]
            if bl_ind.size == 1:
                corr = torch.clone(torch.index_select(corrs, bl_axis, torch.tensor(bl_ind)))
            elif bl_ind.size == 0: # Check for reversed pair
                bl_ind = NP.where((ant_pairs[:, 0] == loop[(i + 1) % loop.size]) & (ant_pairs[:, 1] == loop[i]))[0]
                if bl_ind.size == 0:
                    raise IndexError('Specified antenna pair ({0:0d},{1:0d}) not found in input ant_pairs'.format(loop[i], loop[(i + 1) % loop.size]))
                elif bl_ind.size == 1: # Take Hermitian
                    corr = hermitian(torch.index_select(corrs, bl_axis, torch.tensor(bl_ind)), axes=pol_axes)
                elif bl_ind.size > 1:
                    raise IndexError('{0:0d} indices found for antenna pair ({1:0d},{2:0d}) in input ant_pairs'.format(bl_ind, loop[i], loop[(i + 1) % loop.size]))
            elif bl_ind.size > 1:
                raise IndexError('{0:0d} indices found for antenna pair ({1:0d},{2:0d}) in input ant_pairs'.format(bl_ind, loop[i], loop[(i + 1) % loop.size]))
        
            corr = torch.index_select(corr, bl_axis, torch.tensor([0]))
            corrs_loop += [corr]
        corrs_lol += [corrs_loop]
        
    return corrs_lol

def advariant(corrs_list: Union[List[List[torch.Tensor]], List[torch.Tensor]], pol_axes: Union[List[int], tuple] = (-2, -1)) -> torch.Tensor:
    """
    Construct the advariant from a list of odd-numbered correlations forming a closed odd-edged loop.

    Parameters
    ----------
    corrs_list : List or torch.Tensor of torch.Tensor
        List of odd-numbered correlations forming the edges of a closed odd-edged loop.
    pol_axes : List or tuple of int, optional
        Axes on which the hat operation is applied. Default is (-2, -1).

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
        If the last two dimensions of each torch tensor in corrs_list are not equal.
        If the last two dimensions of each torch tensor in corrs_list are not (1, 1) or (2, 2).
        If the shapes of torch tensors in corrs_list are not identical.

    Notes
    -----
    The advariant is constructed by taking the matrix product of correlations from an odd-numbered list of 
    correlations that forms the edges of a closed odd-edged loop. Every alternate correlation, starting from 
    the second element, undergoes the hat operation, which includes taking the Hermitian conjugate and finding 
    the matrix inverse along the specified polarization axes.

    Examples
    --------
    >>> import torch
    >>> from your_module import advariant
    >>> corrs_list = [
    ...     torch.tensor([[1+2j, 3+4j], [5+6j, 7+8j]]),
    ...     torch.tensor([[9+10j, 11+12j], [13+14j, 15+16j]]),
    ...     torch.tensor([[17+18j, 19+20j], [21+22j, 23+24j]])
    ... ]
    >>> advariant(corrs_list)
    tensor([[ -4.+0.5j,  -5.+1.5j],
            [-12.+8.5j, -13.+9.5j]])
    """

    if not isinstance(corrs_list, (list, torch.Tensor)):
        raise TypeError('Input corrs_list must be a list')
    nedges = len(corrs_list)
    if nedges % 2 == 0:
        raise ValueError('Input corrs_list must be a list made of odd number of elements for an advariant to be constructed')

    if not isinstance(pol_axes, (list, tuple, torch.Tensor)):
        raise TypeError('Input pol_axes must be a list, tuple, or torch tensor')
    if len(pol_axes) != 2:
        raise ValueError('Input pol_axes must be a two-element sequence')
    else:
        for pax in pol_axes:
            if not isinstance(pax, int):
                raise TypeError('Input pol_axes must be a two-element sequence of integers')
    pol_axes = torch.tensor(pol_axes).ravel()

    advar = None
    matgrp = None
    for edgei, corr in enumerate(corrs_list):
        if not isinstance(corr, torch.Tensor):
            raise TypeError('Element {0:0d} of corrs_list must be a torch tensor'.format(edgei))

        if edgei == 0:
            tmpind = torch.where(pol_axes < 0)[0]
            if tmpind.size().numel() > 0:
                pol_axes[tmpind] += corr.ndim  # Convert to a positive value for the polarization axes

            expected_pol_axes = corr.ndim - 2 + torch.arange(2)  # For inverse, they have to be the last two axes
            if not torch.equal(pol_axes, expected_pol_axes):
                raise ValueError('For advariant calculation, pol_axes must be the last two axes because of inherent assumptions about the axes over which matrix multiplication is performed')
        inv_axes = torch.clone(pol_axes)
        if corr.ndim == 1:
            corr = corr[..., None, None]  # shape=(...,n=1,n=1)
        elif corr.ndim >= 2:
            shape2 = torch.tensor(corr.shape[-2:])
            if shape2[0] != shape2[1]:
                raise ValueError('The last two dimensions of each torch tensor that forms the items in the input corrs_list must be equal')
            elif (shape2[0] != 1) and (shape2[0] != 2):
                raise ValueError('The last two dimensions of each torch tensor that forms the items in the input corrs_list must be (1,1) or (2,2) for GL(1,C) and GL(2,C) matrices, respectively')
            if corr.ndim == 2:
                corr = corr[None, :, :]  # shape=(ncorr=1,n=(1/2),n=(1/2))
                inv_axes += 1
        if edgei == 0:
            matgrp = torch.clone(shape2[0])
            advar = torch.clone(corr)
        else:
            if not torch.equal(shape2[0], matgrp):
                raise ValueError('Shape of list element {0:0d} not identical to that of list element 0'.format(edgei))
            if edgei % 2 == 0:
                advar = advar @ corr
            else:
                advar = advar @ hat(corr, axes=inv_axes)

    return advar

def advariants_multiple_loops(corrs_lol: List[List[torch.Tensor]], pol_axes: Union[List[int], tuple] = (-2, -1)) -> torch.Tensor:
    """
    Calculate advariants on multiple loops from a list of lists of odd-numbered correlations forming closed odd-edged loops.

    Parameters
    ----------
    corrs_lol : List of lists of torch.Tensor
        List of lists of odd-numbered correlations forming closed odd-edged loops.
    pol_axes : List or tuple of int, optional
        Axes on which the hat operation is applied. Default is (-2, -1).

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
    list of lists of odd-numbered correlations. The hat operation, which includes taking the Hermitian conjugate and 
    finding the matrix inverse along the specified polarization axes, is applied during the advariant calculation.

    Examples
    --------
    >>> import torch
    >>> from your_module import advariants_multiple_loops
    >>> corrs_lol = [
    ...     [torch.tensor([[1+2j, 3+4j], [5+6j, 7+8j]]), torch.tensor([[9+10j, 11+12j], [13+14j, 15+16j]]), torch.tensor([[17+18j, 19+20j], [21+22j, 23+24j]])],
    ...     [torch.tensor([[25+26j, 27+28j], [29+30j, 31+32j]]), torch.tensor([[33+34j, 35+36j], [37+38j, 39+40j]]), torch.tensor([[41+42j, 43+44j], [45+46j, 47+48j]])]
    ... ]
    >>> advariants_multiple_loops(corrs_lol)
    tensor([[[[ -4. +0.5j,  -5. +1.5j],
              [-12. +8.5j, -13. +9.5j]],
             [[-40.+36.5j, -41.+37.5j],
              [-48.+44.5j, -49.+45.5j]]]])
    """
    if not isinstance(corrs_lol, list):
        raise TypeError('Input corrs_lol must be a list of lists')
    advars_list = []
    for ind, corrs_list in enumerate(corrs_lol):
        advars_list += [advariant(corrs_list, pol_axes=pol_axes)]
    return torch.moveaxis(torch.stack(advars_list), 0, -3).squeeze()  # Move the loop axis to third from the end

def vector_from_advariant(advars):
    """
    Generate 4-vectors from advariants.

    This function takes advariants, which are either 2x2 or 1x1 complex-valued matrices, and generates 4-vectors.

    Parameters:
    advars (torch.Tensor): A torch tensor representing advariants. It can be a 2D array with shape (...,2,2) for GL(2,C) matrices or a 2D array with shape (...,1,1) for GL(1,C) matrices.

    Returns:
    torch.Tensor: A torch tensor representing 4-vectors derived from the advariants. For GL(2,C) matrices, the output has shape (...,4), where the last axis contains the components of the 4-vectors. For GL(1,C) matrices, the input advariants are returned as they already represent 4-vectors.

    Examples:
    >>> import torch
    >>> advars1 = torch.tensor([[1+0j, 2+1j], [2-1j, 3+0j]])  # GL(2,C) matrix
    >>> vector_from_advariant(advars1)
    tensor([[ 2.+0.j, 2.+0.j, -1.+0.j, -1.+0.j]])

    >>> advars2 = torch.tensor([[4+3j]])  # GL(1,C) matrix
    >>> vector_from_advariant(advars2)
    tensor([[4+3j]])
    """
    if not isinstance(advars, torch.Tensor):
        raise TypeError('Input advars must be a torch tensor')
    shape2 = advars.shape[-2:]
    matgrp = shape2[0]

    if (advars.shape[-1] != 2) and (advars.shape[-1] != 1):
        raise ValueError('Input advariant shape incompatible with GL(1,C) or GL(2,C) matrices')
    elif advars.shape[-1] == 2:
        if advars.shape[-1] != advars.shape[-2]:
            raise ValueError('Advariants must be 2x2 matrices in the last two dimensions')
        # Determine Pauli matrices
        pauli_sig_0 = torch.eye(matgrp, dtype=torch.complex128).reshape(tuple(torch.ones(advars.ndim - 2, dtype=int)) + tuple(shape2))  # Psigma_0: shape=(...,n=(1/2),n=(1/2))
        pauli_sig_1 = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128).reshape(tuple(torch.ones(advars.ndim - 2, dtype=int)) + tuple(shape2))  # Psigma_1: shape=(...,n=(1/2),n=(1/2))
        pauli_sig_2 = torch.tensor([[0.0, -1j], [1j, 0.0]], dtype=torch.complex128).reshape(tuple(torch.ones(advars.ndim - 2, dtype=int)) + tuple(shape2))  # Psigma_2: shape=(...,n=(1/2),n=(1/2))
        pauli_sig_3 = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.complex128).reshape(tuple(torch.ones(advars.ndim - 2, dtype=int)) + tuple(shape2))  # Psigma_3: shape=(...,n=(1/2),n=(1/2))

        # Determine components of 4-vectors
        z0 = 0.5 * trace(advars @ pauli_sig_0, axis1=-2, axis2=-1)
        z1 = 0.5 * trace(advars @ pauli_sig_1, axis1=-2, axis2=-1)
        z2 = 0.5 * trace(advars @ pauli_sig_2, axis1=-2, axis2=-1)
        z3 = 0.5 * trace(advars @ pauli_sig_3, axis1=-2, axis2=-1)

        z_4vect = torch.cat([z0[..., None], z1[..., None], z2[..., None], z3[..., None]], dim=-1)  # shape=(...,4)
        return z_4vect
    else:
        if advars.shape[-1] != advars.shape[-2]:
            raise ValueError('Advariants must be 1x1 matrices in the last two dimensions')
        return advars  # GL(1,C) already
    
def trace(input, axis1=0, axis2=1):
    """
    >>> torch.__version__
    '1.9.0.dev20210222+cpu'
    >>> x = torch.arange(1., 10.).view(3, 3)
    >>> x
    tensor([[1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.]])
    >>> torch.trace(x)
    tensor(15.)
    >>> torch.trace(x.view(1, 3, 3))
    Traceback (most recent call last):
    ...
    RuntimeError: trace: expected a matrix, but got tensor with dim 3
    >>> trace(x)
    tensor(15.)
    >>> trace(x.view(3, 3, 1), axis1=0, axis2=1)
    tensor([15.])
    >>> trace(x.view(1, 3, 3), axis1=2, axis2=1)
    tensor([15.])
    >>> trace(x.view(3, 1, 3), axis1=0, axis2=2)
    tensor([15.])
    """
    assert input.shape[axis1] == input.shape[axis2], input.shape

    shape = list(input.shape)
    strides = list(input.stride())
    strides[axis1] += strides[axis2]

    shape[axis2] = 1
    strides[axis2] = 0

    input = torch.as_strided(input, size=shape, stride=strides)
    return input.sum(dim=(axis1, axis2))
    
def minkowski_dot(z1_4v: torch.Tensor, z2_4v: torch.Tensor = None) -> torch.Tensor:
    """
    Compute Minkowski dot products between two sets of 4-vectors.

    This function computes dot products between two sets of 4-vectors using the Minkowski metric. 
    The 4-vectors can either come from advariants or be provided directly.

    Parameters:
    z1_4v : torch.Tensor
        A torch tensor representing the first set of complex 4-vectors. Each 4-vector must have shape (...,4).
    z2_4v : torch.Tensor, optional
        A torch tensor representing the second set of complex 4-vectors. 
        If not provided, the function computes dot products of the first set with itself. 
        Each 4-vector must have the same shape as the 4-vectors in z1_4v.

    Returns:
    torch.Tensor
        A torch tensor containing the Minkowski dot products. 
        If z2_4v is None, the output has shape (..., 2M*(2M+1)//2), where M is the number of complex 4-vectors in z1_4v. 
        If z2_4v is provided, the output has shape (..., 2M*2N), where M and N are the numbers of complex 4-vectors in z1_4v and z2_4v, respectively.

    Examples:
    >>> import torch
    >>> z1_4v = torch.tensor([[ -8.5 +5.j, -8.5 +5.j, 3.5 +3.5j, 4.5 -4.5j],
                              [-44.5+41.j, -44.5+41.j, 3.5 +3.5j, 4.5 -4.5j]])  # A set of 4-vectors
    >>> minkowski_dot(z1_4v)
    tensor([-32.5, -32.5, 8., 8., -32.5, 8., 8., -32.5, -32.5, -32.5])

    >>> z2_4v = torch.tensor([[-80.5+77.j, -80.5+77.j, 3.5 +3.5j, 4.5 -4.5j]])  # Another set of 4-vectors
    >>> minkowski_dot(z1_4v, z2_4v)
    tensor([-32.5, 8., -32.5, 8., 8., -32.5, 8., -32.5])
    """
    if not isinstance(z1_4v, torch.Tensor):
        raise TypeError('Input z1_4v must be a torch tensor')
    shape1 = torch.tensor(z1_4v.shape)
    if shape1[-1] != 4:
        raise ValueError('The dimension of last axis in input z1_4v must equal 4 to be a valid 4-vector')
    
    metric = torch.tensor([[1,0,0,0], 
                           [0,-1,0,0], 
                           [0,0,-1,0], 
                           [0,0,0,-1]], dtype=torch.float64).reshape(-1,4) # Minkowski metric
    
    x1_4v = z1_4v.real # shape=(...,M,4)
    y1_4v = z1_4v.imag # shape=(...,M,4)
    stack1_4v = torch.cat([x1_4v, y1_4v], dim=-2) # shape=(...,2M,4)
    if z2_4v is None:
        stack2_4v = stack1_4v.clone() # shape=(...,2M,4)
    else:
        if not isinstance(z2_4v, torch.Tensor):
            raise TypeError('Input z2_4v must be a torch tensor')
        shape2 = torch.tensor(z2_4v.shape)
        if not torch.equal(shape1[-1], shape2[-1]):
            raise ValueError('The dimension of last axis in inputs z1_4v and z2_4v must match')
        x2_4v = z2_4v.real # shape=(...,N,4)
        y2_4v = z2_4v.imag # shape=(...,N,4)
        stack2_4v = torch.cat([x2_4v, y2_4v], dim=-2) # shape=(...,2N,4)

    mdp = torch.einsum('...ij,jk,...lk->...il', stack1_4v, metric, stack2_4v) # shape=(...2M,2N)
    if z2_4v is None: # Return only the upper diagonal
        upperind = torch.triu_indices(mdp.shape[-1], mdp.shape[-1])
        upperind_raveled = ravel_multi_index(upperind, mdp.shape[-2:])
        mdp = mdp.reshape(mdp.shape[:-2]+(-1,))
        mdp = mdp[...,upperind_raveled]
    else:
        mdp = mdp.reshape(mdp.shape[:-2]+(-1,))       
    
    return mdp    

def ravel_multi_index(multi_index, dims):
    """
    Converts a tuple of index arrays into an array of flat indices, applying the given shape.

    Parameters:
    multi_index (tuple of torch.Tensor): A tuple of integer arrays, one for each dimension.
    dims (torch.Size or tuple): The shape of the array.

    Returns:
    torch.Tensor: An array of flat indices.

    Examples:
    >>> multi_index = (torch.tensor([1, 2]), torch.tensor([3, 4]))
    >>> dims = (3, 5)
    >>> ravel_multi_index(multi_index, dims)
    tensor([ 8, 14])
    """
    if not all(isinstance(idx, torch.Tensor) for idx in multi_index):
        raise TypeError('All elements of multi_index must be torch tensors')
    if not isinstance(dims, (tuple, torch.Size)):
        raise TypeError('dims must be a tuple or torch.Size')

    dims = torch.tensor(dims)
    strides = torch.cumprod(torch.cat((torch.tensor([1]), dims[:-1])), dim=0)
    return sum(idx * stride for idx, stride in zip(multi_index, strides))

def complete_minkowski_dots(z4v: torch.Tensor) -> torch.Tensor:
    """
    Compute the complete set of Minkowski inner products between a given set of Lorentz 4-vectors formed from advariants.

    This function computes the complete set of Minkowski inner products between a given set of Lorentz 4-vectors formed from advariants.
    The 4-vectors are partitioned into two groups: the first two vectors are considered as basis vectors,
    and the rest are treated as vectors for which inner products with the basis vectors are computed.

    Parameters:
    z4v : torch.Tensor
        A torch tensor representing the set of Lorentz complex 4-vectors of shape (..., N, 4). 

    Returns:
    torch.Tensor
        A torch tensor containing the complete set of Minkowski inner products.
        The output has shape (..., 8N-6), where N is the number of 4-vectors.

    Examples:
    >>> import torch
    >>> z4v = torch.tensor([[[ -8.5 +5.j, -8.5 +5.j, 3.5 +3.5j, 4.5 -4.5j],
                             [-44.5+41.j, -44.5+41.j, 3.5 +3.5j, 4.5 -4.5j],
                             [-80.5+77.j, -80.5+77.j, 3.5 +3.5j, 4.5 -4.5j]]])  # A set of Lorentz 4-vectors
    >>> complete_minkowski_dots(z4v)
    tensor([[-32.5, -32.5, 8., 8., -32.5, 8., 8., -32.5, -32.5, -32.5, -32.5, 8., 
            -32.5, 8., 8., -32.5, 8., -32.5]])
    """

    if not isinstance(z4v, torch.Tensor):
        raise TypeError('Input z1_4v must be a torch tensor')
    if z4v.ndim == 1:
        z4v = z4v.reshape(1,-1)
    # z4v = z4v.to(torch.complex64)
    inshape = torch.tensor(z4v.shape)
    z4v_basis = z4v[...,:2,:] # Choose first two complex 4-vectors for the basis
    z4v_rest = z4v[...,2:,:] # Choose first two complex 4-vectors for the rest
    
    mdp_basis_basis = minkowski_dot(z4v_basis)
    mdp_basis_rest = minkowski_dot(z4v_basis, z4v_rest)
    
    mdp = torch.cat([mdp_basis_basis, mdp_basis_rest], dim=-1)
    return mdp

def remove_scaling_factor_minkoski_dots(mdp: torch.Tensor, wts: torch.Tensor = None) -> torch.Tensor:
    """
    Remove a common scaling factor from the input array mdp by using the weights array wts.

    Parameters:
    mdp : torch.Tensor
        The input array representing Minkowski dot products. 
        It has shape (...,N).
    wts : torch.Tensor, optional
        The weights array used to compute the weighted L2 norm of the input array. 
        If None, the weights are considered as ones. They must be broadcastable 
        to mdp

    Returns:
    torch.Tensor
        The output array after removing the scaling factor.
        It has shape (...,N).

    Examples:
    >>> import torch
    >>> mdp = torch.tensor([[-32.5, -32.5, 8., 8., -32.5, 8., 8., -32.5, 
                             -32.5, -32.5, -32.5, 8., -32.5, 8., 8., 
                             -32.5, 8., -32.5]])  # Input Minkowski dot products
    >>> remove_scaling_factor_minkoski_dots(mdp)
    tensor([[-0.3088, -0.3088,  0.0760,  0.0760, -0.3088,  0.0760,  0.0760, -0.3088, 
            -0.3088, -0.3088, -0.3088,  0.0760, -0.3088,  0.0760,  0.0760, 
            -0.3088,  0.0760, -0.3088]])
    """
    if wts is None:
        wts = torch.ones_like(mdp)

    # Remove scaling factor
    ci = mdp / torch.sqrt(torch.sum((wts*mdp)**2, dim=-1, keepdim=True))

    return ci


################################################################################

def hermitian(inparr: torch.Tensor, axes: Optional[Union[List[int], Tuple[int, int], torch.Tensor]] = (-2, -1)) -> torch.Tensor:
    """
    Return the Hermitian of the input array along specified axes.

    Parameters
    ----------
    inparr : torch.Tensor
        Input array to be Hermitian-transposed.
    axes : {tuple, list, torch.Tensor}, optional
        Two-element sequence denoting which two axes are to be Hermitian-transposed.
        Default is (-2, -1).

    Returns
    -------
    torch.Tensor
        Hermitian-transposed array.

    Raises
    ------
    TypeError
        If inparr is not a torch tensor or if axes is not a list, tuple, or torch tensor.
    ValueError
        If axes is not a two-element list, tuple, or torch tensor or if the two entries in axes are the same.

    Notes
    -----
    The Hermitian of an array is obtained by swapping the specified axes and taking the complex conjugate.

    Examples
    --------
    >>> import torch
    >>> arr = torch.tensor([[1, 2 + 1j], [3 - 2j, 4]], dtype=torch.complex64)
    >>> hermitian(arr, axes=(0, 1))
    tensor([[1.-0.j, 3.+2.j],
            [2.-1.j, 4.-0.j]])
    """
    # axes denotes which two axes are to be Hermitian-transposed
    if not isinstance(inparr, torch.Tensor):
        raise TypeError('Input array inparr must be a torch tensor')
    if inparr.ndim == 1:
        inparr = inparr.reshape(1, -1)

    if axes is None:
        axes = torch.tensor([-2, -1])
    if not isinstance(axes, (list, tuple, torch.Tensor)):
        raise TypeError('Input axes must be a list, tuple, or torch tensor')
    axes = torch.tensor(axes).flatten()
    if axes.size(0) != 2:
        raise ValueError('Input axes must be a two-element list, tuple, or torch tensor')
    negind = torch.where(axes < 0)[0]
    if negind.size(0) > 0:
        axes[negind] += inparr.ndim  # Convert negative axis numbers to positive
    if axes[0] == axes[1]:
        raise ValueError('The two entries in axes cannot be the same')

    return inparr.transpose(axes[0], axes[1]).conj()

################################################################################

def hat(inparr: torch.Tensor, axes: Optional[Union[List[int], Tuple[int, int], torch.Tensor]] = None) -> torch.Tensor:
    """
    Compute the inverse of the Hermitian operation along specified axes (hat operation).

    Parameters
    ----------
    inparr : torch.Tensor
        Input array to be inversed with the Hermitian operation.
    axes : Union[list, tuple, torch.Tensor], optional
        Two-element sequence denoting which two axes are to be Hermitian-transposed.
        If None, defaults to torch.tensor([-2, -1]).

    Returns
    -------
    torch.Tensor
        Result of the hat operation, the inverse of the Hermitian-transposed array.

    Raises
    ------
    TypeError
        If inparr is not a torch tensor.
    ValueError
        If axes is not a two-element list, tuple, or torch tensor or if the two entries in axes are the same.

    Notes
    -----
    The hat operation is the inverse of the Hermitian operation. It involves computing the Hermitian-transposed array,
    taking the inverse along the last two axes using torch.linalg.inv(), and optionally rearranging the axes.

    Examples
    --------
    >>> import torch
    >>> arr = torch.tensor([[1, 2 + 1j], [3 - 2j, 4]], dtype=torch.complex64)
    >>> hat_result = hat(arr, axes=(0, 1))
    """

    if not isinstance(inparr, torch.Tensor):
        raise TypeError('Input array inparr must be a torch tensor')
    if inparr.ndim == 1:
        inparr = inparr.reshape(1, -1)

    if axes is None:
        axes = torch.tensor([-2, -1])
    if not isinstance(axes, (list, tuple, torch.Tensor)):
        raise TypeError('Input axes must be a list, tuple, or torch tensor')
    axes = torch.tensor(axes).flatten()
    if axes.size(0) != 2:
        raise ValueError('Input axes must be a two-element list, tuple, or torch tensor')
    negind = torch.where(axes < 0)[0]
    if negind.size(0) > 0:
        axes[negind] += inparr.ndim  # Convert negative axis numbers to positive
    if axes[0] == axes[1]:
        raise ValueError('The two entries in axes cannot be the same')

    inparr_H = hermitian(inparr, axes=axes)
    invaxes = inparr.ndim - 2 + torch.arange(2)  # For inverse, they must be the last two axes because of torch.linalg.inv() requirements
    if not torch.equal(torch.sort(axes)[0], invaxes):  # the axes are not at the end, so move them for taking inverse
        inparr_H = inparr_H.permute(*[i for i in range(inparr_H.ndim) if i not in axes], *invaxes)
    if inparr_H.shape[-1] != inparr_H.shape[-2]:
        raise ValueError('The axes of inversion must be square in shape')
    inparr_IH = torch.linalg.inv(inparr_H)
    if not torch.equal(torch.sort(axes)[0], invaxes):  # the axes were moved to the end, so move them back
        inparr_IH = inparr_IH.permute(*[i for i in range(inparr_IH.ndim) if i not in invaxes], *axes)

    return inparr_IH
