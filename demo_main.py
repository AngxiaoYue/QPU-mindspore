import json
import os
import sys

import mindspore.context as context
import mindspore.dataset as ds
import mindspore.nn as nn
import numpy as np
import yaml
from mindspore import Model
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.nn.metrics import Accuracy
from demo_dataset import GetDatasetGenerator
from demo_model import QLSTM, RLSTM
from mindspore.train.callback import (Callback, CheckpointConfig, LossMonitor,
                                      ModelCheckpoint)

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# custom callback function
class StepLossAccInfo(Callback):
    def __init__(self, model, eval_dataset, steps_res, log_path):
        self.model = model
        self.eval_dataset = eval_dataset
        self.steps_res = steps_res
        self.log_path = log_path

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        cur_step = (cur_epoch-1)*1177 + cb_params.cur_step_num
        print("epoch:{}, batch:{}/1177, losse: {}".format(cur_epoch, cb_params.cur_step_num % 1177, str(cb_params.net_outputs)))
        self.steps_res["loss_value"].append(str(cb_params.net_outputs))
        if cur_step % 1177 == 0:
            acc = self.model.eval(self.eval_dataset, dataset_sink_mode=False)
            self.steps_res["epoch"].append(str(cur_epoch))
            self.steps_res["acc"].append(acc["Accuracy"])
            json_object = json.dumps(self.steps_res, indent=3)
            with open(self.log_path + "/res.json", "w") as outfile:
                outfile.write(json_object)

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def train(config):

    ## prepare for dataset
    model_name = config["net"]
    num_epochs = int(config['num_epochs'])
    batch_size = int(config['batch_size'])
    learning_rate = float(config['learning_rate'])
    weight_decay = float(config['weight_decay'])
    data_dir = config[config['dataset']]
    model_path = config['model_path']
    log_path = config['logdir']
    val_data_dir = data_dir
    if config['dataset'] == 'ntu':
        num_joints = 25
        num_cls = 60
    else:
        raise ValueError

    train_label_path = os.path.join(data_dir, 'train_label.pkl')
    val_label_path = os.path.join(val_data_dir, 'val_label.pkl')
    train_edge_path = os.path.join(data_dir, 'train_data_rel.npy') if config['use_edge'] else None
    test_edge_path = os.path.join(val_data_dir, 'val_data_rel.npy') if config['use_edge'] else None
    if 'edge_only' in config and config['edge_only']:
        print(os.path.join(data_dir, 'train_data_rel.npy'))
        traindata = GetDatasetGenerator(os.path.join(data_dir, 'train_data_rel.npy'), train_label_path, None, num_samples=-1,
                                        mmap=True, num_frames=config['data_param']['num_frames'])
  
        testdata = GetDatasetGenerator(os.path.join(val_data_dir, 'val_data_rel.npy'), val_label_path, None, num_samples=-1,
                                        mmap=True, num_frames=config['data_param']['num_frames'])
    else:
        traindata = GetDatasetGenerator(os.path.join(data_dir, 'train_data.npy'), train_label_path, train_edge_path, num_samples=-1,
                                        mmap=True, num_frames=config['data_param']['num_frames'])
        testdata = GetDatasetGenerator(os.path.join(val_data_dir, 'val_data.npy'), val_label_path, test_edge_path, num_samples=-1,
                                        mmap=True, num_frames=config['data_param']['num_frames'])
    # print(len(traindata)) # 37646
    trainloader = ds.GeneratorDataset(traindata, ["data", "label"], shuffle=True, num_parallel_workers=4)
    trainloader = trainloader.batch(batch_size)
    testloader = ds.GeneratorDataset(testdata, ["data", "label"], shuffle=True, num_parallel_workers=4)
    testloader = testloader.batch(batch_size)


    ## get model
    net = QLSTM(config['in_channels'], num_joints, config['data_param']['num_frames'], num_cls, config)
    # net = RLSTM(config['in_channels'], num_joints, config['data_param']['num_frames'], num_cls, config)
    net_opt = nn.Adam(net.trainable_params(), learning_rate=learning_rate, weight_decay=weight_decay)
    net_loss = nn.CrossEntropyLoss()
    net = Model(net, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

 
    config_ck = CheckpointConfig(save_checkpoint_steps=1177, keep_checkpoint_max=16)
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_{}".format(model_name), directory=model_path, config=config_ck)
    steps_res = {"epoch": [], "loss_value": [], "acc": []}
    step_loss_acc_info = StepLossAccInfo(net , testloader, steps_res, log_path)
    net.train(num_epochs, trainloader, callbacks=[ckpoint_cb, LossMonitor(1177), step_loss_acc_info], dataset_sink_mode=False)
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError('Please enter config file path.')
    
    # Read configs
    configFile = sys.argv[1] # path of configfile
    with open(configFile, 'r') as f:
        config = yaml.safe_load(f)
    
    # Train
    train(config)