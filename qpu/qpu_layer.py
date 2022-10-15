import math

import mindspore as ms
import mindspore.ops as ops
import numpy as np
from mindspore import Parameter, Tensor
from mindspore import dtype as mstype
from mindspore import nn
from mindspore.common.initializer import (Normal, Uniform, XavierUniform,
                                          _calculate_fan_in_and_fan_out,
                                          initializer)
from qpu.qpu_op import qpu_linear, quaternion_normalize


class QPU(nn.Cell):
    """Quaternion product units. Apply weights on scalar part. Then perform chained Hamilton product.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(QPU, self).__init__()
        self.in_features = in_features // 4 
        self.out_features = out_features // 4 
        self.weight = Parameter(initializer('XavierUniform', [self.out_features, self.in_features], mstype.float32))
        if bias:
            fan_in, fan_out = _calculate_fan_in_and_fan_out(self.weight.shape)
            a = math.sqrt(6 / (fan_in + fan_out))
            self.bias = Parameter(initializer(Uniform(a), [self.out_features], mstype.float32))

    def construct(self, input):
        output = qpu_linear(input, self.weight, self.bias)
        output = quaternion_normalize(output, dim=-1)
        return output

class AngleAxisMap(nn.Cell):
    """
    change the scalar part of a quaternion
    """
    def __init__(self, dim=-1, rinv=False):
        super(AngleAxisMap, self).__init__()
        self.dim = dim
        self.rinv = rinv

    def construct(self, input):
        split = ops.Split(self.dim , 4)
        min = Tensor(-1+1e-6, ms.float32)
        max = Tensor(1-1e-6, ms.float32)
        r, i, j, k = split(input)
        r = ops.acos(ops.clip_by_value(r, min, max))
        if self.rinv:
            return r
        sinTheta = ops.sin(r)
        i /= sinTheta
        j /= sinTheta
        k /= sinTheta
        concat_op = ops.Concat(-1)

        return concat_op((r, i, j, k))

class KeepRealPart(nn.Cell):
    """
    Keep scalar part of a quaternion
    """
    def __init__(self, dim=-1):
        super(KeepRealPart, self).__init__()
        self.dim = dim

    def construct(self, input):
        split = ops.Split(self.dim , 4)
        r = split(input)[0]
        return r


