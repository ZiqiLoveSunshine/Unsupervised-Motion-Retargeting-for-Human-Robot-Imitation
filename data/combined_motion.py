from torch.utils.data import Dataset
import copy
from data.motion_dataset import MotionData
import os
import numpy as np
import torch
from data.bvh_parser import BVH_file
from data import get_test_set
import sys
from option_parser import get_std_bvh, get_std_bvh_test


class MixedData0(Dataset):
    """
    Mixed data for many skeletons but one topologies
    """
    def __init__(self, args, motions, skeleton_idx):
        super(MixedData0, self).__init__()
        self.motions = motions
        self.motions_reverse = torch.tensor(self.motions.numpy()[..., ::-1].copy())
        self.skeleton_idx = skeleton_idx
        self.length = motions.shape[0]
        self.args = args

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        if self.args.data_augment == 0 or torch.rand(1) < 0.5:
            return [self.motions[item], self.skeleton_idx[item]]
        else:
            return [self.motions_reverse[item], self.skeleton_idx[item]]


class MixedData(Dataset):
    """
    data_gruop_num * 2 * samples
    """
    def __init__(self, args, datasets_groups):
        # device = torch.device(args.cuda_device if (torch.cuda.is_available()) else 'cpu')
        self.final_data = []
        self.length = 0
        self.offsets = []
        # self.joint_topologies = []
        self.character_num = len(datasets_groups)
        seed = 19260817
        all_datas = []
        # for datasets in datasets_groups:
        for i, dataset in enumerate(datasets_groups):
            new_args = copy.copy(args)
            new_args.data_augment = 0
            new_args.dataset = dataset

            all_datas.append(MotionData(new_args))
            file = BVH_file(get_std_bvh(dataset=dataset))
            if i == 0:
                self.joint_topology = file.topology
            new_offset = file.get_normalize_offset()
            new_offset = torch.tensor(new_offset, dtype=torch.float)
            new_offset = new_offset.reshape((1,) + new_offset.shape)
            self.offsets.append(new_offset)
        self.offsets = torch.cat(self.offsets, dim=0)
        # self.offsets = self.offsets.to(device)

        pt = 0
        for datasets in all_datas:
            skeleton_idx = []
            # print(datasets.data.shape)
            skeleton_idx += [pt]*len(datasets)
            pt += 1
            # print(skeleton_idx)
            if self.length != 0 and self.length != len(skeleton_idx):
                self.length = min(self.length, len(skeleton_idx))
            else:
                self.length = len(skeleton_idx)
                # print("this len: ", self.length)
            self.final_data.append(MixedData0(args, datasets.data, skeleton_idx))


    def __len__(self):
        return self.length

    def __getitem__(self, item):
        res = []
        for data in self.final_data:
            res.append(data[item])
        return res


class TestData(Dataset):
    def __init__(self, args, characters):
        self.character_num = len(characters)
        self.characters = characters
        self.file_list = get_test_set()
        self.joint_topologies = []
        self.offsets = []
        self.args = args
        self.device = torch.device(args.cuda_device)

        for i, dataset in enumerate(characters):
            file = BVH_file(get_std_bvh_test(dataset=dataset))
            if i == 0:
                self.joint_topology = file.topology
            new_offset = file.get_normalize_offset()
            new_offset = torch.tensor(new_offset, dtype=torch.float)
            new_offset = new_offset.reshape((1,) + new_offset.shape)
            self.offsets.append(new_offset)
        self.offsets = torch.cat(self.offsets, dim=0)

    def __getitem__(self, item):
        res = []
        bad_flag = 0
        ref_shape = None
        res_group = []
        for i, character in enumerate(self.characters):
            new_motion = self.get_item(i, item)
            if new_motion is not None:
                new_motion = new_motion.reshape((1, ) + new_motion.shape)
                ref_shape = new_motion
            res_group.append(new_motion)

        if ref_shape is None:
            print('Bad at {}'.format(item))
            return None
        for j in range(len(self.characters)):
            if res_group[j] is None:
                bad_flag = 1
                res_group[j] = torch.zeros_like(ref_shape)
        if bad_flag:
            print('Bad at {}'.format(item))

        res_group = torch.cat(res_group, dim=0)
        res.append(res_group)
        res.append(list(range(len(self.characters))))
        res.append(self.item_len)
        return res

    def __len__(self):
        return len(self.file_list)

    def get_item(self, gid, id):
        character = self.characters[gid]
        path = './datasets/test_set/{}/'.format(character)
        if isinstance(id, int):
            file = path + self.file_list[id]
        elif isinstance(id, str):
            file = id
        else:
            raise Exception('Wrong input file type')
        print(file)
        if not os.path.exists(file):
            raise Exception('Cannot find file')
        file = BVH_file(file)
        motion = file.to_dataset()
        # print("combined motion, motion shape: {}, motion: {}" .format(motion.shape, motion))
        if motion.shape[0] > self.args.window_size:
            motion = motion[::2, ...]
        self.item_len = motion.shape[0]
        motion = self.get_windows(motion)
        print("motion shape: ", motion.shape)
        return motion.to(self.device)

    def get_windows(self, motion):
        new_windows = []
        # print("get window, motion shape: ", motion.shape)
        if motion.shape[0] % self.args.window_size > self.args.window_size//2:
            add = self.args.window_size - motion.shape[0] % self.args.window_size
            temp = np.empty((add, motion.shape[1], motion.shape[2]))
            temp[:,:,:] = motion[-1,:,:]
            motion = np.vstack([motion,temp])
        # print("get window, motion shape now: ", motion.shape)
        step_size = self.args.window_size
        window_size = self.args.window_size
        n_window = motion.shape[0] // step_size

        for i in range(n_window):
            begin = i * step_size
            end = begin + window_size

            new = motion[begin:end, :]
            new = new[np.newaxis, ...]

            new_window = torch.tensor(new, dtype=torch.float32)
            new_windows.append(new_window)

        return torch.cat(new_windows)

