# QPU Layer

## QPU
```
class QPU(in_channels, out_channels, bias=True)
```
aking N quaternions as its inputs, the QPU first applies quaternion power operation to each input, scaling their
rotation angles and rotation axes, respectively. Then, it applies a chain of Hamilton products to merge the weighted
quaternions and output a rotation accordingly. The parameters of the QPU consists of the scaling coefficients and the
bias introduced to the rotation angles.

## Parameters

## Inputs

## Outputs

## Supported Platforms

## Examples
```python

```
