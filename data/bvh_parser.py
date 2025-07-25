import sys
import torch
# sys.path.append("../utils")
import utils.BVH_mod as BVH
import numpy as np
from utils.Quaternions import Quaternions
from utils.skeleton import build_edge_topology
# from option_parser import get_std_bvh
from data.bvh_writer import write_bvh

"""
1.
Specify the joints that you want to use in training and test. Other joints will be discarded.
Please start with root joint, then left leg chain, right leg chain, head chain, left shoulder chain and right shoulder chain.
See the examples below.
"""
corps_name_1 = ['Pelvis', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']
# corps_name_2 = ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'LeftToe_End', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'RightToe_End', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 'HeadTop_End', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']
corps_name_2 = ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase',  'Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 'HeadTop_End', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']
corps_name_3 = ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase',  'Spine', 'Spine1', 'Spine2', 'Neck', 'Neck1','Head', 'HeadTop_End', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']
corps_name_4 = ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase',  'Spine', 'Spine1', 'Spine1_split','Spine2', 'Neck', 'Head', 'HeadTop_End', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']
"""
2.
Specify five end effectors' name.
Please follow the same order as in 1.
"""
ee_name_1 = ['LeftToeBase', 'RightToeBase', 'Head', 'LeftHand', 'RightHand']
ee_name_my = ['LeftToeBase', 'RightToeBase', 'HeadTop_End', 'LeftHand', 'RightHand']



corps_names = [corps_name_1, corps_name_2, corps_name_3 , corps_name_4]
ee_names = [ee_name_1, ee_name_my, ee_name_1 , ee_name_my]
"""
3.
Add previously added corps_name and ee_name at the end of the two above lists.
"""
# corps_names.append(corps_name_example)
# ee_names.append(ee_name_example)


class BVH_file:
    def __init__(self, file_path=None, args=None, dataset=None, new_root=None):
        # if file_path is None:
        #     file_path = get_std_bvh(dataset=dataset)
        self.anim, self._names, self.frametime = BVH.load(file_path)
        
        # print('BVH_parser animation', self.anim[0])
        # print('BVH_parser names', self._names)
        # print('BVH_parser frametime', self.frametime)
        if new_root is not None:
            self.set_new_root(new_root)
        self.skeleton_type = -1
        self.edges = []
        self.edge_mat = []
        self.edge_num = 0
        self._topology = None
        self.ee_length = []

        for i, name in enumerate(self._names):
            if ':' in name:
                name = name[name.find(':') + 1:]
                self._names[i] = name

        full_fill = [1] * len(corps_names)
        for i, ref_names in enumerate(corps_names):
            for ref_name in ref_names:
                if ref_name not in self._names:
                    full_fill[i] = 0
                    break

        # if full_fill[3]:
        #     self.skeleton_type = 3
        # else:
        for i, _ in enumerate(full_fill):
            if full_fill[i]:
                self.skeleton_type = i
                break

        if 'Spine1_split' in self._names:
            self.skeleton_type = 3
        """
        4. 
        Here, you need to assign self.skeleton_type the corresponding index of your own dataset in corps_names or ee_names list.
        You can use self._names, which contains the joints name in original bvh file, to write your own if statement.
        """
        # if ...:
        #     self.skeleton_type = 11

        if self.skeleton_type == -1:
            print(self._names)
            raise Exception('Unknown skeleton')

        if self.skeleton_type == 0:
            self.set_new_root(1)

        self.details = [i for i, name in enumerate(self._names) if name not in corps_names[self.skeleton_type]]
        self.joint_num = self.anim.shape[1]
        self.corps = []
        self.simplified_name = []
        self.simplify_map = {}
        self.inverse_simplify_map = {}

        for name in corps_names[self.skeleton_type]:
            for j in range(self.anim.shape[1]):
                if name == self._names[j]:
                    self.corps.append(j)
                    break

        if len(self.corps) != len(corps_names[self.skeleton_type]):
            for i in self.corps: print(self._names[i], end=' ')
            print(self.corps, self.skeleton_type, len(self.corps), sep='\n')
            raise Exception('Problem in file', file_path)

        self.ee_id = []
        for i in ee_names[self.skeleton_type]:
            self.ee_id.append(corps_names[self.skeleton_type].index(i))

        self.joint_num_simplify = len(self.corps)
        for i, j in enumerate(self.corps):
            self.simplify_map[j] = i
            self.inverse_simplify_map[i] = j
            self.simplified_name.append(self._names[j])
        self.inverse_simplify_map[0] = -1
        for i in range(self.anim.shape[1]):
            if i in self.details:
                self.simplify_map[i] = -1

        self.edges = build_edge_topology(self.topology, self.offset)
        # print("file path:{}, skeleton type: {}".format(file_path, self.skeleton_type))
    

        
    @property
    def topology(self):
        if self._topology is None:
            self._topology = self.anim.parents[self.corps].copy()
            for i in range(self._topology.shape[0]):
                if i >= 1: self._topology[i] = self.simplify_map[self._topology[i]]
            self._topology = tuple(self._topology)
        return self._topology

    def get_ee_id(self):
        return self.ee_id

    #for each frame, we put the rotation and position together.
    def to_numpy(self, quater=False, edge=True):
        rotations = self.anim.rotations[:, self.corps, :]
        if quater:
            rotations = Quaternions.from_euler(np.radians(rotations)).qs
            positions = self.anim.positions[:, 0, :]
        else:
            positions = self.anim.positions[:, 0, :]
        
        if edge:
            index = []
            for e in self.edges:
                index.append(e[0])
            rotations = rotations[:, index, :]

        rotations = rotations.reshape(rotations.shape[0], -1)

        return np.concatenate((rotations, positions), axis=1)

    def to_dataset(self):
        rotations = self.anim.rotations[:, self.corps, :]
        positions = self.anim.positions[:, 0, :]
        offsets = self.offset
        dis_ref = self.calculate_ref_dis()
        world_pos = self.forward(rotations, positions, offsets)
        return world_pos/dis_ref



    def calculate_ref_dis(self):
        for i in range(len(self.names)):
            if self.names[i] == "HeadTop_End":
                begin_num = i
            if self.names[i] == "LeftToeBase":
                end_num = i
        begin_y = 0
        end_y = 0
        while begin_num != -1:
            begin_y += self.offset[begin_num][1]
            begin_num = self.topology[begin_num]
        while end_num != -1:
            end_y += self.offset[end_num][1]
            end_num = self.topology[end_num]
        return begin_y - end_y
        # print(begin_y)
        # print(end_y)

    def forward(self, rotation, position, offset, order='xyz', ignore_root_offset = True):
        result = np.empty(rotation.shape[:-1] + (3, ))
        # print("result: ", result.shape)

        transform = self.transform_from_euler(rotation, order)
        offset_matrix = self.transform_offset_matrix(position.shape[0],offset, ignore_root_offset)
        position_matrix = self.transform_position_matrix(position)
        parent = np.zeros_like(transform)
    #     print("transform:", transform.shape)
    #     print("offset_matrix: ", offset_matrix.shape)

    #     print("position:", position_matrix.shape)
        for i, pi in enumerate(self.topology):
            if pi == -1:
            # we ignore the root offset or there will be problems
                localtoworld_root = np.matmul(offset_matrix[:,0,...],position_matrix)
                parent[:, 0,...] = np.matmul(localtoworld_root, transform[:, 0, ...])
    #             print(temp)
                result[:, 0, 0] = localtoworld_root[...,0,3]
                result[:, 0, 1] = localtoworld_root[...,1,3]
                result[:, 0, 2] = localtoworld_root[...,2,3]
    #             print(result)
                continue
    #         print("parent: {}, now: {}".format(pi, i))
            localtoworld = np.matmul(parent[:,pi,...],offset_matrix[:, i, ...])
            parent[:, i,...] = np.matmul(localtoworld, transform[:, i, ...])
            result[:, i, 0] = localtoworld[...,0,3]
            result[:, i, 1] = localtoworld[...,1,3]
            result[:, i, 2] = localtoworld[...,2,3]
        return result


    def transform_from_euler(self, rotation, order):
        rotation = rotation / 180 * np.pi
        transform = np.matmul(self.transform_from_axis(rotation[..., 1], order[1]),
                                self.transform_from_axis(rotation[..., 2], order[2]))
        transform = np.matmul(self.transform_from_axis(rotation[..., 0], order[0]), transform)
        return transform
    
    def transform_from_axis(self, euler, axis):
        transform = np.zeros(euler.shape[0:3] + (4, 4))
        cos = np.cos(euler)
        sin = np.sin(euler)
    #     transform[..., :, :] = transform[..., :, :] = 0.0
        for i in range(4):
            transform[..., i, i] = 1.0
        
    #     print(transform)
        if axis == 'x':
            transform[..., 1, 1] = transform[..., 2, 2] = cos
            transform[..., 1, 2] = -sin
            transform[..., 2, 1] = sin
        if axis == 'y':
            transform[..., 0, 0] = transform[..., 2, 2] = cos
            transform[..., 0, 2] = sin
            transform[..., 2, 0] = -sin
        if axis == 'z':
            transform[..., 0, 0] = transform[..., 1, 1] = cos
            transform[..., 0, 1] = -sin
            transform[..., 1, 0] = sin

        return transform

    def transform_offset_matrix(self, frame, offset, ignore_root_offset):
        offset_matrix = np.zeros([frame, offset.shape[0], 4, 4])
        for i in range(4):
            offset_matrix[..., i, i] = 1.0
    #     print(offset_matrix)
        offset_matrix[..., :3, 3] = offset
        if ignore_root_offset:
            offset_matrix[:,0,...] = np.eye(4)
    #     print(offset_matrix)
        return offset_matrix

    def transform_position_matrix(self, position):
        position_matrix = np.zeros([position.shape[0], 4, 4])
        for i in range(4):
            position_matrix[..., i, i] = 1.0
    #     print(offset_matrix)
        position_matrix[:, :3, 3] = position
        return position_matrix


    def to_tensor(self, quater=False, edge=True):
        res = self.to_numpy(quater, edge)
        res = torch.tensor(res, dtype=torch.float)
        res = res.permute(1, 0)
        res = res.reshape((-1, res.shape[-1]))
        return res

    def get_position(self):
        positions = self.anim.positions
        positions = positions[:, self.corps, :]
        return positions

    @property
    def offset(self):
        return self.anim.offsets[self.corps]
    
    def get_normalize_offset(self):
        return np.linalg.norm(self.offset, axis = 1)/self.calculate_ref_dis()

    @property
    def names(self):
        return self.simplified_name

    def get_height(self):
        offset = self.offset
        topo = self.topology

        res = 0
        p = self.ee_id[0]
        while p != 0:
            res += np.dot(offset[p], offset[p]) ** 0.5
            p = topo[p]

        p = self.ee_id[2]
        while p != 0:
            res += np.dot(offset[p], offset[p]) ** 0.5
            p = topo[p]

        return res

    def write(self, file_path):
        motion = self.to_numpy(quater=False, edge=False)
        rotations = motion[..., :-3].reshape(motion.shape[0], -1, 3)
        positions = motion[..., -3:]
        write_bvh(self.topology, self.offset, rotations, positions, self.names, 1.0/30, 'xyz', file_path)
        
    def set_new_root(self, new_root):
        euler = torch.tensor(self.anim.rotations[:, 0, :], dtype=torch.float)
        transform = self.transform_from_euler(euler, 'xyz')
        offset = torch.tensor(self.anim.offsets[new_root], dtype=torch.float)
        new_pos = torch.matmul(transform, offset)
        new_pos = new_pos.numpy() + self.anim.positions[:, 0, :]
        
        self.anim.offsets[0] = -self.anim.offsets[new_root]
        self.anim.offsets[new_root] = np.zeros((3, ))
        self.anim.positions[:, new_root, :] = new_pos
        rot0 = Quaternions.from_euler(np.radians(self.anim.rotations[:, 0, :]), order='xyz')
        rot1 = Quaternions.from_euler(np.radians(self.anim.rotations[:, new_root, :]), order='xyz')
        new_rot1 = rot0 * rot1
        new_rot0 = (-rot1)
        new_rot0 = np.degrees(new_rot0.euler())
        new_rot1 = np.degrees(new_rot1.euler())
        self.anim.rotations[:, 0, :] = new_rot0
        self.anim.rotations[:, new_root, :] = new_rot1

        new_seq = []
        vis = [0] * self.anim.rotations.shape[1]
        new_idx = [-1] * len(vis)
        new_parent = [0] * len(vis)

        def relabel(x):
            nonlocal new_seq, vis, new_idx, new_parent
            new_idx[x] = len(new_seq)
            new_seq.append(x)
            vis[x] = 1
            for y in range(len(vis)):
                if not vis[y] and (self.anim.parents[x] == y or self.anim.parents[y] == x):
                    relabel(y)
                    new_parent[new_idx[y]] = new_idx[x]

        relabel(new_root)
        self.anim.rotations = self.anim.rotations[:, new_seq, :]
        self.anim.offsets = self.anim.offsets[new_seq]
        names = self._names.copy()
        for i, j in enumerate(new_seq):
            self._names[i] = names[j]
        self.anim.parents = np.array(new_parent, dtype=np.int)
