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
**x** (Tensor) - Tensor of shape ```(∗,in_channels)```. The in_channels in Args should be equal to in_channels in Inputs. 

Note: in_channels = ```4 * _ ```, we should cat quaternions' r, i, j, k parts to get the input.

For example, ``` [batch, _ , 4, num_joint] -> [batch, _ , 4*num_joint] -> QPU(). ``` Num\_joint is the number of rotation nodes in the application scenario.

## Outputs
Tensor of shape ```(∗,out_channels)```.

## Supported Platforms
```GPU```, ```CPU```

## Examples
```python
>>> context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU") # or CPU
>>> x = Tensor(np.array([[[0.6836, 0.6947, -0.2234, 0.0125], [0.4961, 0.5503, -0.0914, 0.6653]], [[0.8631, 0.2327, -0.4219, 0.1512], [0.7958, -0.0660, -0.5379, -0.2703]]]), mindspore.float32)
>>> batch, num_joint, _ = x.shape
>>> x = ops.transpose(x, (0, 2, 1))
>>> x = ops.reshape(x, (batch, 4 * num_joint))
>>> net = QPU(4 * num_joint, 16)
>>> output = net(x)
>>>ptint(output.shape)
(2, 16)
```
For systematic example, please refer to the ```demo_main.py``` to run the demo models.
