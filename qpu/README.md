# QPU Layer

## QPU
```
class QPU(in_channels, out_channels, bias=True)
```
Taking N quaternions as its inputs, the QPU first applies quaternion power operation to each input, scaling their
rotation angles and rotation axes, respectively. Then, it applies a chain of Hamilton products to merge the weighted
quaternions and output a rotation accordingly. The parameters of the QPU consists of the scaling coefficients and the
bias introduced to the rotation angles.

## Parameters
in_channels (int) – The number of channels in the input space.
out_channels (int) – The number of channels in the output space.

## Inputs
x (Tensor) - Tensor of shape (∗,in_channels). The in_channels in Args should be equal to in_channels in Inputs. 

Note: in_channels = ```4 * _ ```, we should cat quaternions' r, i, j, k parts to get the input.

For example, ``` [batch, _ , 4, num_joint] -> [batch, _ , 4*num_joint] -> QPU(). ``` Num\_joint is the number of rotation nodes in the application scenario.

## Outputs
Tensor of shape (∗,out_channels).

## Supported Platforms
```GPU```, ```CPU```

## Examples
```python

```
