import torch
import torch.nn as nn
from models.skeleton_gnn import Node2EdgeLayer, Edge2NodeLayer
from models.enc_and_dec import calculate_adj_matrix

"""
Inputs:
    node_feats - Tensor with node features of shape [batch_size, num_joints, c_in]-N*23*3   - normalized position information
    edge_feats - Tensor with edge features of shape [batch_size, num_offset + one_base, features]- N*23*1  - normalized offset information
    adj_matrix - Batch of adjacency matrices of the graph. If there is an edge from i to j, adj_matrix[b,i,j]=1 else 0.
                Supports directed edges by non-symmetric matrices. Assumes to already have added the identity connections.
                Shape: [batch_size, num_nodes, num_nodes]

    input - form [batch_size, num_joint, c_in] - N * num_joint *(x, y, z)
"""
class Discriminator(nn.Module):
    def __init__(self, args, topology):
        super(Discriminator, self).__init__()
        self.args = args
        self.len = len(topology)
        self.device = torch.device(args.cuda_device if (torch.cuda.is_available()) else 'cpu')
        self.c_in_position = args.position_feature
        self.c_in_offset = args.offset_feature
        self.adj_matrix = calculate_adj_matrix(topology).to(self.device)
        self.layer1_out = 4
        self.layer2_out = 8
        self.layer3_out = 16
        self.layer4_out = 8
        # network module
        self.layers = nn.ModuleList()
        self.layer1 = nn.ModuleList()
        self.layer2 = nn.ModuleList()
        self.layer3 = nn.ModuleList()
        self.layer4 = nn.ModuleList()
        self.layer1.append(nn.Conv1d(self.c_in_position*self.len, self.layer1_out*self.len, padding = 1, kernel_size=4, stride=2))
        self.layer1.append(nn.LeakyReLU())
        self.layer1.append(nn.Conv1d(self.c_in_offset*self.len, self.layer1_out*self.len, padding = 1, kernel_size=4, stride=2))
        self.layer1.append(nn.LeakyReLU())
        self.layer2.append(Node2EdgeLayer(self.layer1_out, self.layer1_out, self.layer2_out))
        self.layer2.append(nn.LeakyReLU())
        self.layer2.append(Edge2NodeLayer(self.layer2_out, self.layer1_out, self.layer2_out))
        self.layer2.append(nn.LeakyReLU())
        self.layer3.append(nn.Conv1d(self.layer2_out*self.len, self.layer3_out*self.len, padding = 1, kernel_size=4, stride=2))
        self.layer3.append(nn.LeakyReLU())
        self.layer3.append(nn.Conv1d(self.layer2_out*self.len, self.layer3_out*self.len, padding = 1, kernel_size=4, stride=2))
        self.layer3.append(nn.LeakyReLU())
        self.layer4.append(Node2EdgeLayer(self.layer3_out, self.layer3_out, self.layer4_out))
        self.layer4.append(nn.LeakyReLU())
        self.layer4.append(Edge2NodeLayer(self.layer4_out, self.layer3_out, self.layer4_out))
        self.layer4.append(nn.LeakyReLU())
        # network layers
        self.layers.append(self.layer1)
        self.layers.append(self.layer2)
        self.layers.append(self.layer3)
        self.layers.append(self.layer4)
        self.linear = nn.ModuleList()
        self.linear.append(nn.Linear(self.len*self.layer4_out*2, 64))
        # self.linear.append(nn.Linear(self.len*(self.layer4_out*2)*int(args.window_size/4), 64))
        self.linear.append(nn.LeakyReLU())
        self.linear.append(nn.Linear(64, 1))
        # 
        # self.linear.append(nn.Sigmoid())

    def forward(self, input):
        offset = input[...,-1].unsqueeze(-1)
        input = input[..., :-1]
        for i, layer in enumerate(self.layers):
            if i%2 ==0:
                for j, network in enumerate(layer):
                    if j == 0: # input
                        batch_size, frame, num_joint, _ = input.shape
                        input = input.permute(0, 2, 3, 1) # (batch_size, num_joint, feature_in, frame)
                        input = input.reshape((batch_size, -1, frame)) # (batch_size, num_joint*feature_in, frame)
                        input = network(input) # (batch_size, num_joint * feature_out, ?)
                        input = input.permute(0, 2, 1)  # (batch_size, frame/2, num_joint * feature_out)
                        feature_out = int(input.shape[-1])
                        # print(input.shape)
                        input = input.reshape((batch_size, -1, num_joint, feature_out//num_joint)) # (batch_size, frame/2, num_joint, feature_out)
                        input = layer[1](input)
                    elif j == 2: # offset
                        batch_size, frame, num_joint, _ = offset.shape
                        offset = offset.permute(0, 2, 3, 1) # (batch_size, num_joint, feature_in, frame)
                        offset = offset.reshape((batch_size, -1, frame)) # (batch_size, num_joint*feature_in, frame)
                        offset = network(offset) # (batch_size, num_joint * feature_out, ?)
                        offset = offset.permute(0, 2, 1)  # (batch_size, frame/2, num_joint * feature_out)
                        feature_out = int(offset.shape[-1])
                        offset = offset.reshape((batch_size, -1, num_joint, feature_out//num_joint)) # (batch_size, frame/2, num_joint, feature_out)
                        offset = layer[-1](offset)
            else:
                # print(i, layer)
                batch_size, frame, _,__ = input.shape
                adj_matrix = torch.cat(batch_size*frame*[self.adj_matrix]).reshape((batch_size, frame, self.len, self.len))
                offset = layer[0](input, offset, adj_matrix) # (batch_size, frame/2, num_joint, offset_out)
                offset = layer[1](offset)
                input = layer[2](input, offset, adj_matrix) # (batch_size, frame/2, num_joint, position_out)
                input = layer[3](input)
        input = torch.concat([offset, input], dim = -1) # (batch_size, frame/2, num_joint, fea_xyz + fea_offset)
        batch_size, frame, _,__ = input.shape
        input = input.reshape((batch_size, frame, -1)) # (batch_size, frame/2, num_joint * xyz + offset)
        if self.args.dis_linear:
            for net in self.linear:
                input = net(input)
        return input