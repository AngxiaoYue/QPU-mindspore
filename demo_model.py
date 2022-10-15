import mindspore.ops as ops
import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import nn
from mindspore.common.initializer import Normal

np.set_printoptions(precision=4, threshold=np.inf, suppress=True)
from qpu.qpu_layer import QPU, AngleAxisMap


class RLSTM(nn.Cell):
    def __init__(self, in_channels, num_joints, num_frames, num_cls, config):
        super(RLSTM, self).__init__()
        self.num_joints = num_joints
        self.num_frames = num_frames
        self.feat_dim = 256
        self.mlp = nn.SequentialCell([
            nn.Dense(self.num_joints * 4, self.feat_dim, Normal(0.02), Normal(0.02)),
            nn.ReLU(),
            nn.Dense(self.feat_dim, self.feat_dim, Normal(0.02), Normal(0.02)),
            nn.ReLU(),
        ])

        self.lstm = nn.LSTM(input_size=self.feat_dim, hidden_size=self.feat_dim, num_layers=1)

        self.classifier = nn.SequentialCell([
            nn.Dense(self.feat_dim, self.feat_dim, Normal(0.02), Normal(0.02)),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Dense(self.feat_dim, num_cls, Normal(0.02), Normal(0.02))
        ])

    def construct(self, x):
        """
        Args:
            x: Tensor(B, 4, F, J, M) (batch_size, 4, num_frame, num_joint, max_body)
        """
        batch_size = x.shape[0]
        num_person=x.shape[-1]
        x = ops.transpose(x, (0, 4, 2, 1, 3)) # B, M, F, 4, J
        x = ops.reshape(x, (batch_size * num_person * self.num_frames, self.num_joints * 4))
        x = self.mlp(x)  # B*M*F, C*4
        x = ops.reshape(x, (batch_size*num_person, self.num_frames, -1)) # (B*M, F, C*4)
        x = ops.transpose(x, (1, 0, 2)) # (F, B*M, C*4)
        h0 = Tensor(np.zeros((1, batch_size*num_person, self.feat_dim)), dtype=mstype.float32)
        c0 = Tensor(np.zeros((1, batch_size*num_person, self.feat_dim)), dtype=mstype.float32)
        output, (hn, cn) = self.lstm(x, (h0, c0))  # out: (F, B*M, C)
        output = ops.reshape(ops.transpose(output, (1, 0, 2)), (batch_size, num_person * self.num_frames, -1))
        x = ops.mean(output, 1)
        x = self.classifier(x)
        return x

class QLSTM(nn.Cell):
    def __init__(self, in_channels, num_joints, num_frames, num_cls, config):
        super(QLSTM, self).__init__()
        self.config = config
        self.num_joints = num_joints
        self.num_frames = num_frames
        self.feat_dim = 256
        if 'rinv' in config and config['rinv']:
            self.mlp = nn.SequentialCell([
                QPU(self.num_joints * 4, self.feat_dim), # 64 qpu neurons
                QPU(self.feat_dim, self.feat_dim*4), # 256 qpu neurons
                AngleAxisMap(dim=-1,rinv=True)
            ])
        else:
            self.mlp = nn.SequentialCell([
                QPU(self.num_joints * 4, self.feat_dim),
                QPU(self.feat_dim, self.feat_dim),
                AngleAxisMap(dim=-1,rinv=False)
            ])

        self.lstm = nn.LSTM(input_size=self.feat_dim, hidden_size=self.feat_dim, num_layers=1)

        self.classifier = nn.SequentialCell([
            nn.Dense(self.feat_dim, self.feat_dim, Normal(0.02), Normal(0.02)),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Dense(self.feat_dim, num_cls, Normal(0.02), Normal(0.02))
        ])

    def construct(self, x):
        """
        Args:
            x: Tensor(B, 4, F, J, M)
        """
        batch_size = x.shape[0]
        num_person = x.shape[-1]
        x = ops.transpose(x, (0, 4, 2, 1, 3)) # B, M, F, 4, J
        x = ops.reshape(x, (batch_size * num_person * self.num_frames, self.num_joints * 4))
        x = self.mlp(x)  # B*M*F, C*4
        x = ops.reshape(x, (batch_size*num_person, self.num_frames, -1)) # (B*M, F, C*4)
        x = ops.transpose(x, (1, 0, 2)) # (F, B*M, C*4)
        h0 = Tensor(np.zeros((1, batch_size*num_person, self.feat_dim)), dtype=mstype.float32)
        c0 = Tensor(np.zeros((1, batch_size*num_person, self.feat_dim)), dtype=mstype.float32)
        output, (hn, cn) = self.lstm(x, (h0, c0))  # out: (F, B*M, C)
        output = ops.reshape(ops.transpose(output, (1, 0, 2)), (batch_size, num_person * self.num_frames, -1))
        x = ops.mean(output, 1)
        x = self.classifier(x)
        return x


