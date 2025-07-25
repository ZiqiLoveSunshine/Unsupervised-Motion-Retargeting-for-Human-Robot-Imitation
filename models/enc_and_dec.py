import torch
from torch import nn
from models.skeleton_gnn import Node2EdgeLayer, Edge2NodeLayer

"""
Inputs:
    node_feats - Tensor with node features of shape [batch_size, num_joints, c_in]-N*23*3   - normalized position information
    edge_feats - Tensor with edge features of shape [batch_size, num_offset, features]- N*23*1  - normalized offset information
    adj_matrix - Batch of adjacency matrices of the graph. If there is an edge from i to j, adj_matrix[b,i,j]=1 else 0.
                Supports directed edges by non-symmetric matrices. Assumes to already have added the identity connections.
                Shape: [batch_size, num_nodes, num_nodes]

    input - form [batch_size, num_joint, c_in] - N * num_joint *(x, y, z)
    offset - form [batch_size, num_joint, joint_len] - N * num_joint * offset
"""
def calculate_adj_matrix(topology):
    adj = torch.zeros((len(topology), len(topology)))
    for i,j in enumerate(topology):
        if i == 0: #the root not in
            continue
        adj[j,i] = 1
    return adj

class Encoder(nn.Module):
    def __init__(self, args, topology):
        super(Encoder, self).__init__()
        self.len = len(topology)
        self.device = torch.device(args.cuda_device if (torch.cuda.is_available()) else 'cpu')
        self.c_in_position = args.position_feature
        self.c_in_offset = args.offset_feature
        self.adj_matrix = calculate_adj_matrix(topology).to(self.device)
        self.layer1_out = 4
        self.layer2_out = 8
        self.layer3_out = 16
        # network module
        self.layers = nn.ModuleList()
        self.layer1 = nn.ModuleList()
        self.layer2 = nn.ModuleList()
        self.layer3 = nn.ModuleList()
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
        # network layers
        self.layers.append(self.layer1)
        self.layers.append(self.layer2)
        self.layers.append(self.layer3)

    def forward(self, input, offset):
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
        return input

class Decoder(nn.Module):
    def __init__(self, args, enc: Encoder):
        super(Decoder, self).__init__()
        self.enc = enc
        self.len = self.enc.len
        self.c_in_position = self.enc.layer3_out
        self.c_in_offset = args.offset_feature
        self.adj_matrix = self.enc.adj_matrix
        self.layer1_out = 8
        self.layer2_out = 4
        self.c_out_position = args.position_feature
        self.c_out_offset = args.offset_feature
        # network module
        self.layers = nn.ModuleList()
        self.layer1 = nn.ModuleList()
        self.layer2 = nn.ModuleList()
        self.layer3 = nn.ModuleList()
        self.layer1.append(nn.Upsample(scale_factor=(self.layer1_out/self.c_in_position,2), mode="bilinear"))
        self.layer1.append(nn.Upsample(scale_factor=(self.layer1_out/self.c_in_offset,2), mode="bilinear"))
        self.layer2.append(Node2EdgeLayer(self.layer1_out, self.layer1_out, self.layer2_out))
        self.layer2.append(nn.LeakyReLU())
        self.layer2.append(Edge2NodeLayer(self.layer2_out, self.layer1_out, self.layer2_out))
        self.layer2.append(nn.LeakyReLU())
        self.layer3.append(nn.ConvTranspose1d(self.layer2_out*self.len, self.c_out_position*self.len, padding = 1, kernel_size=4, stride=2))
        self.layer3.append(nn.LeakyReLU())
        self.layer3.append(nn.ConvTranspose1d(self.layer2_out*self.len, self.c_out_offset*self.len, padding = 1, kernel_size=4, stride=2))
        self.layer3.append(nn.LeakyReLU())
        # network layers
        self.layers.append(self.layer1)
        self.layers.append(self.layer2)
        self.layers.append(self.layer3)

    def forward(self, input, offset):
        for i, layer in enumerate(self.layers):
            if i ==0:
                for j, network in enumerate(layer):
                    if j == 0: # input
                        batch_size, frame, num_joint, _ = input.shape
                        input = input.permute(0, 2, 3, 1) # (batch_size, num_joint, feature_in, frame)
                        input = network(input) # (batch_size, num_joint, feature_out, frame *2)
                        input = input.permute(0, 3, 1, 2)  # (batch_size, frame*2, num_joint, feature_out)
                    elif j == 1: # offset
                        batch_size, frame, num_joint, _ = offset.shape
                        offset = offset.permute(0, 2, 3, 1) # (batch_size, num_joint, feature_in, frame)
                        offset = network(offset) # (batch_size, num_joint, feature_out, frame *2)
                        offset = offset.permute(0, 3, 1, 2)  # (batch_size, frame*2, num_joint, feature_out)
                # print("decoder, input: {}, offset: {}".format(input.shape, offset.shape))
            elif i == 1:
                # print(i, layer)
                batch_size, frame, _,__ = input.shape
                adj_matrix = torch.cat(batch_size*frame*[self.adj_matrix]).reshape((batch_size, frame, self.len, self.len))
                # print("offset avant: ", offset[0,0])
                offset = layer[0](input, offset, adj_matrix) # (batch_size, frame/2, num_joint, offset_out)
                offset = layer[1](offset)
                input = layer[2](input, offset, adj_matrix) # (batch_size, frame/2, num_joint, position_out)
                input = layer[3](input)
                # print("offset apres: ", offset[0,0])
                # print("input apres: ", input[0, 0])
            elif i == 2:
                for j, network in enumerate(layer):
                    if j == 0: # input
                        batch_size, frame, num_joint, _ = input.shape
                        input = input.permute(0, 2, 3, 1) # (batch_size, num_joint, feature_in, frame)
                        input = input.reshape((batch_size, -1, frame)) # (batch_size, num_joint*feature_in, frame)
                        input = network(input) # (batch_size, num_joint * feature_out, ?)
                        input = input.permute(0, 2, 1)  # (batch_size, frame*2, num_joint * feature_out)
                        feature_out = int(input.shape[-1])
                        # print(input.shape)
                        input = input.reshape((batch_size, -1, num_joint, feature_out//num_joint)) # (batch_size, frame*2, num_joint, feature_out)
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
                # print("after conv input apres: ", input[0,0])
        return input

