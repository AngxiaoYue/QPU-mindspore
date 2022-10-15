# QPU-mindspore
Quaternion Product Units (QPU) implemented by mindspore. 
**The QPU can be used as a network layer which is scalable and useful**. The detailed README is placed in ``qpu`` directory,

## Data preparation
We use the the [NTU RGB+D dataset](https://github.com/shahroudy/NTURGB-D). We recommend you to put them under the ```raw_data``` directory. Please refer to their respective project pages for permission and downloading. Use the following commands to convert to our data formats.
```
python data/scripts/ntu_gendata.py <dir of raw data> --mode xyz --bench <xview or xsub>
python data/scripts/ntu_gendata.py <dir of raw data> --mode qrel --bench <xview or xsub>
python data/scripts/gen_edge.py <dir of qrel data> --dataset ntu
```
Generated data is in ```<dir of raw data>/NTU-RGB-D/qrel```.

## Train and Test
We choose **RMLP_LSTM** and **QMLP_LSTM** as our demo models. we provide ```.yaml``` file for convenient train and test.
```
python train.py configs/<config name>.yaml
```
Check the path, especially the path in .yaml file.

## Performance under mindspore framework (accuracy)
|  QMLP_LSTM  | RMLP_LSTM  |
|  ----  | ----  |
|  0.7428 |  0.7252 |


