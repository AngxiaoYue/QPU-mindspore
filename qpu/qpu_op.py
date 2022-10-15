import mindspore as ms
import mindspore.ops as ops
import numpy as np
from mindspore import Tensor, nn


def qpu_linear(input, weight, bias):
    """
    input: (*, C_in * 4) -> (*, C_out * 4)
    """
    in_channels = input.shape[-1] 
    in_channels = in_channels // 4 
    out_channels = weight.shape[0] 
    split = ops.Split(-1, 4)
    expand_dims = ops.ExpandDims()
    r, i, j, k = split(expand_dims(input, -2))
    r, i, j, k = quaternion_power_bias(r, i, j, k, weight, bias) 
    quaternionRemoveZeros = QuaternionRemoveZeros()
    r, i, j, k = quaternionRemoveZeros(r, i, j, k)
    r, i, j, k = quaternion_chained_prod(r, i, j, k, -1)
    concat_op = ops.Concat(-1)
    return concat_op((r, i, j, k)) 


class  QuaternionRemoveZeros(nn.Cell):
    def __init__(self):
        super(QuaternionRemoveZeros, self).__init__()
 
    def construct(self, r, i, j, k):
        norm = r**2+ i**2+ j**2+ k**2
        index = norm == 0
        r[index] = 1
        return r,i,j,k

    def bprop(self, r, i, j, k, out, dout):
        norm = r**2+ i**2+ j**2+ k**2
        index = norm == 0
        gr, gi, gj, gk = dout
        gr[index] = 0
        gi[index] = 0
        gj[index] = 0
        gk[index] = 0
        return gr, gi, gj, gk
    

def quaternion_normalize(input, dim):
    """ Normalize quaternion
    """
    in_channels = input.shape[dim] // 4
    split = ops.Split(-1, 4)
    r, i, j, k = split(input)
    norm = ops.sqrt(r**2 + i**2 + j**2 + k**2 + 1e-12)
    r = r / norm
    i = i / norm
    j = j / norm
    k = k / norm
    concat_op = ops.Concat(-1)
    return concat_op((r, i, j, k))


def quaternion_power_bias(r, i, j, k, weight, bias):
    """
    r, i, j, k: (*, 1, C_in)
    weight: (C_out, C_in)
    bias: (C_out)
    return: [cos(w * (acos(r) + bias)), sin(w * (acos(r) + bias)) v / |v|]
    """
    ## Compute new theta
    sqrt = ops.Sqrt()
    norm_v = sqrt(i**2 + j**2 + k**2 + 1e-12)
    min = Tensor(-1+1e-6, ms.float32)
    max = Tensor(1-1e-6, ms.float32)
    theta = ops.acos(ops.clip_by_value(r, min, max))
    if bias is not None:
        expand_dims = ops.ExpandDims()
        theta = theta + expand_dims(bias, -1)
    theta = weight * theta 
    
    mul = ops.sin(theta) / norm_v
    r = ops.cos(theta)
    i = i * mul
    j = j * mul
    k = k * mul
    return r, i, j, k


def quaternion_power(r, i, j, k, w):
    """
    r, i, j, k: (..., C_in, ...)
    w: (..., C_in, ...)
    return: [cos(w * acos(r)), sin(w * acos(r)) v / |v|]
    """    
    ## Compute new theta
    sqrt = ops.Sqrt()
    norm_v = sqrt(i**2 + j**2 + k**2 + 1e-12)
    min = Tensor(-1+1e-6, ms.float32)
    max = Tensor(1-1e-6, ms.float32)
    theta = w * ops.acos(ops.clip_by_value(r, min, max))
    ## Compute new quaternion
    r = ops.cos(theta)
    mul = ops.sin(theta) / norm_v
    i = i * mul
    j = j * mul
    k = k * mul
    return r, i, j, k


def quaternion_chained_prod_loop(r_input, i_input, j_input, k_input, dim=-1):
    """
    Chained quaternion product along a dimension (for loop)
    Hamilton product:
    a1 a2 - b1 b2 - c1 c2 - d1 d2 
    + ( a1 b2 + b1 a2 + c1 d2 − d1 c2 ) i
    + ( a1 c2 − b1 d2 + c1 a2 + d1 b2 ) j 
    + ( a1 d2 + b1 c2 − c1 b2 + d1 a2 ) k 
    """
    seq_len = r_input.shape[dim]
    r_out, i_out, j_out, k_out = r_input[..., 0], i_input[..., 0], j_input[..., 0], k_input[..., 0]
    for i in range(1, seq_len):
        r_out, i_out, j_out, k_out = hamilton_product_chunk(r_out, i_out, j_out, k_out, r_input[..., i], i_input[..., i], j_input[..., i], k_input[..., i])

    return r_out, i_out, j_out, k_out

def unfold(tensor):
    """
    Function has the same effect as torch.Tensor.unfold(dim, 2, 2).
    """
    batch_size, C_out, channel = tensor.shape
    if channel % 2 == 1:
        return ops.reshape(tensor[..., :-1], (batch_size, C_out, channel // 2, 2))
    else:
        return ops.reshape(tensor, (batch_size, C_out, channel // 2, 2))
        

def quaternion_chained_prod(r_input, i_input, j_input, k_input, dim, last=None):
    """
    Chained quaternion product along a dimension (recursive)
    Hamilton product:
    a1 a2 - b1 b2 - c1 c2 - d1 d2 
    + ( a1 b2 + b1 a2 + c1 d2 − d1 c2 ) i
    + ( a1 c2 − b1 d2 + c1 a2 + d1 b2 ) j 
    + ( a1 d2 + b1 c2 − c1 b2 + d1 a2 ) k 
    """
    expand_dims = ops.ExpandDims()
    squeeze = ops.Squeeze(dim)
    channel = r_input.shape[dim]
    if channel == 1:
        return squeeze(r_input), squeeze(i_input), squeeze(j_input), squeeze(k_input)
    else:
        ## Split into pair(0) and odd(1)
        r_out, i_out, j_out, k_out = unfold(r_input), unfold(i_input), unfold(j_input), unfold(k_input)

        r_pair, r_odd = r_out[..., 0], r_out[..., 1]
        i_pair, i_odd = i_out[..., 0], i_out[..., 1]
        j_pair, j_odd = j_out[..., 0], j_out[..., 1]
        k_pair, k_odd = k_out[..., 0], k_out[..., 1]

        ## pair * odd
        r_out, i_out, j_out, k_out = hamilton_product_chunk(r_pair, i_pair, j_pair, k_pair, r_odd, i_odd, j_odd, k_odd)

        ## Multiply last
        if channel % 2 == 1:
             last = (r_input[..., -1], i_input[..., -1], j_input[..., -1], k_input[..., -1])

        if r_out.shape[dim] % 2 == 1 and last is not None:
             concat_op = ops.Concat(dim)
             r_out = concat_op((r_out,expand_dims(last[0], dim)))
             i_out = concat_op((i_out,expand_dims(last[1], dim)))
             j_out = concat_op((j_out,expand_dims(last[2], dim)))
             k_out = concat_op((k_out,expand_dims(last[3], dim)))
             last = None
        ## Recursion
        r_out, i_out, j_out, k_out = quaternion_chained_prod(r_out, i_out, j_out, k_out, dim, last)
        return r_out, i_out, j_out, k_out


def hamilton_product_chunk(r1, i1, j1, k1, r2, i2, j2, k2):
    """
    Hamilton product
    a1 a2 - b1 b2 - c1 c2 - d1 d2 
    + ( a1 b2 + b1 a2 + c1 d2 − d1 c2 ) i
    + ( a1 c2 − b1 d2 + c1 a2 + d1 b2 ) j 
    + ( a1 d2 + b1 c2 − c1 b2 + d1 a2 ) k 
    """

    r_out, i_out, j_out, k_out = r1 * r2 - i1 * i2 - j1 * j2 - k1 * k2, \
                                 r1 * i2 + i1 * r2 + j1 * k2 - k1 * j2, \
                                 r1 * j2 - i1 * k2 + j1 * r2 + k1 * i2, \
                                 r1 * k2 + i1 * j2 - j1 * i2 + k1 * r2
    return r_out, i_out, j_out, k_out


