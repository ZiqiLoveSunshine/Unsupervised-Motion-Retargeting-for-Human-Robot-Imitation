
import torch
from torch import nn


"""
Inputs:
    node_feats - Tensor with node features of shape [batch_size, frame, num_joints, c_in] - batch_size *N*23*3   - normalized position information
    edge_feats - Tensor with edge features of shape [batch_size, frame, num_offset, features]- batch_size *N*23*1  - normalized offset information
    adj_matrix - Batch of adjacency matrices of the graph. If there is an edge from i to j, adj_matrix[b,f,i,j]=1 else 0.
                Supports directed edges by non-symmetric matrices. Assumes to already have added the identity connections.
                Shape: [batch_size, frame, num_nodes, num_nodes]
"""

class Node2NodeLayer(nn.Module):
    
    def __init__(self, c_in_node, c_out_node):
        super().__init__()
        self.parent_linear = nn.Linear(c_in_node, c_out_node)
        self.recurrent = nn.Linear(c_in_node, c_out_node)
        self.children_linear = nn.Linear(c_in_node, c_out_node)
        self.c_in_node = c_in_node
        self.c_out_node = c_out_node
        
    def forward(self, node_features, adj_matrix, aggr='mean'):
        # We assume that the diagonal of adj_matrix is empty
        
        # Mean aggregation (try sum maybe ?)
        num_children = adj_matrix.sum(dim=-1, keepdims=True) 
        num_parents = adj_matrix.transpose(-1, -2).sum(dim=-1, keepdims=True)
        num_neighbours = num_children + num_parents + 1
        batch_size, frame, num_joints, _ = node_features.shape # (batch_size, frame, num_joints, c_in)
        
        # Parents
        parent_features = self.parent_linear(node_features)
        parent_features = parent_features.reshape(-1, num_joints, self.c_out_node) # (batch_size*frame, num_joints, c_in)
        parent_features = torch.bmm(adj_matrix.transpose(-1, -2).reshape(-1, num_joints, num_joints), parent_features)
        parent_features = parent_features.reshape(batch_size, frame, num_joints, self.c_out_node)
        # print('parent features: ', parent_features.shape)
        # Recurrent
        recurrent_features = self.recurrent(node_features)
        
        # Children
        children_features = self.children_linear(node_features)
        children_features = children_features.reshape(-1, num_joints, self.c_out_node) # (batch_size*frame, num_joints, c_in)
        children_features = torch.bmm(adj_matrix.reshape(-1, num_joints, num_joints), children_features)
        children_features = children_features.reshape(batch_size, frame, num_joints, self.c_out_node)

        # Mean features (possible improvements : other aggregation, weighted sum)
        if aggr=='mean':
            node_features = (parent_features + children_features + recurrent_features) / num_neighbours
        elif aggr=='sum':
            node_features = parent_features + children_features + recurrent_features
        
        # Possibly other activation function
        # node_features = torch.sigmoid(node_features)
        
        # shape (batch_size, num_joint, c_out_node)
        return node_features

class Node2EdgeLayer(nn.Module):
    
    def __init__(self, c_in_node, c_in_edge, c_out_edge):
        super().__init__()
        
        self.parent_linear = nn.Linear(c_in_node, c_out_edge)
        self.recurrent = nn.Linear(c_in_edge, c_out_edge)
        self.recurrent.weight = nn.Parameter(self.recurrent.weight*10)
        self.children_linear = nn.Linear(c_in_node, c_out_edge)
        
        self.c_in_node = c_in_node
        self.c_in_edge = c_in_edge
        self.c_out_edge = c_out_edge
        
    def forward(self, node_features, edge_features, adj_matrix, aggr='mean'):
        
        # node_features [batch_size, num_joints, c_in_node]
        # edge_features [batch_size, num_joints, c_in_edge]
        
        # Each node has only one parent edge (parent_edge_feautres)
        # Nodes can have several children edges bmm(adj, children_edge_features)
        
        # For each node, the edge of same index is the parent edge
        # For each node, the adjacency matrix gives the indices of children nodes
        
        # Mean aggregation (try sum maybe ?)
        num_children = adj_matrix.sum(dim=-1, keepdims=True)
        num_parents = adj_matrix.transpose(-1, -2).sum(dim=-1, keepdims=True)
        num_neighbours = num_children + num_parents + 1
        batch_size, frame, num_joints, _ = node_features.shape
        
        # Children
        children_features = self.children_linear(node_features)
        # print("children feature: ", children_features[0,0])
        # Recurrent
        recurrent_features = self.recurrent(edge_features)
        # print("recurrent feature: ", recurrent_features[0,0])
        # Parents
        parent_features = self.parent_linear(node_features)
        parent_features = parent_features.reshape(-1, num_joints, self.c_out_edge) # (batch_size*frame, num_joints, c_out)
        parent_features = torch.bmm(adj_matrix.transpose(-1, -2).reshape(-1, num_joints, num_joints), parent_features)
        parent_features = parent_features.reshape(batch_size, frame, num_joints, self.c_out_edge)
        # print("parent feature: ", parent_features[0,0])
        # Mean features (possible improvements : other aggregation, weighted sum)
        if aggr=='mean':
            edge_features = (parent_features + children_features + recurrent_features) / 3
        elif aggr=='sum':
            edge_features = parent_features + children_features + recurrent_features
        
        # Possibly other activation function
        # edge_features = torch.sigmoid(edge_features)
        
        # shape (batch_size, frame, num_joint, c_out_edge)
        return edge_features

class Edge2NodeLayer(nn.Module):
    def __init__(self, c_in_edge, c_in_node, c_out_node):
        super().__init__()
        self.c_in_edge = c_in_edge
        self.c_in_node = c_in_node
        self.c_out_node = c_out_node
        
        self.parent_linear = nn.Linear(c_in_edge, c_out_node)
        # self.parent_linear.weight = nn.Parameter(self.parent_linear.weight*10)
        self.recurrent = nn.Linear(c_in_node, c_out_node)
        self.children_linear = nn.Linear(c_in_edge, c_out_node)
        # self.children_linear.weight = nn.Parameter(self.children_linear.weight*10)
        
    def forward(self, node_features, edge_features, adj_matrix, aggr='mean'):
        
        # Mean aggregation (try sum maybe ?)
        num_children = adj_matrix.sum(dim=-1, keepdims=True)
        num_parents = adj_matrix.transpose(-1, -2).sum(dim=-1, keepdims=True)
        num_neighbours = num_children + num_parents + 1
        batch_size, frame, num_joints, _ = node_features.shape
        
        # Children
        children_features = self.children_linear(edge_features)
        children_features = children_features.reshape(-1, num_joints, self.c_out_node) # (batch_size*frame, num_joints, c_out)
        children_features = torch.bmm(adj_matrix.reshape(-1, num_joints, num_joints), children_features)
        children_features = children_features.reshape(batch_size, frame, num_joints, self.c_out_node) *5
        
        # Recurrent
        recurrent_features = self.recurrent(node_features)
        
        # Parents
        parent_features = self.parent_linear(edge_features)*5
        
        # Mean features (possible improvements : other aggregation, weighted sum)
        if aggr=='mean':
            res = (parent_features + children_features + recurrent_features) / num_neighbours
        elif aggr=='sum':
            res = parent_features + children_features + recurrent_features
        
        # Possibly other activation function
        # res = torch.sigmoid(res)
        

        # shape (batch_size, num_joint, c_out_node)
        return res