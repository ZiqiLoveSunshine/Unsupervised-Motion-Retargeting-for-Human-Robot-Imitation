from torch.utils.data import Dataset
import numpy as np
import torch


class MotionData(Dataset):
    """
    Clip long dataset into fixed length window for batched training
    each data is a 2d tensor with shape (Joint_num*3) * Time
    """
    def __init__(self, args):
        super(MotionData, self).__init__()
        name = args.dataset
        file_path = './datasets/train_set/npy/{}.npy'.format(name)

        if args.debug:
            file_path = file_path[:-4] + '_debug' + file_path[-4:]

        print('load from file {}'.format(file_path))
        self.total_frame = 0
        # self.std_bvh = get_std_bvh(args)
        self.args = args
        self.data = []
        self.motion_length = []
        motions = np.load(file_path, allow_pickle=True)
        motions = list(motions)

        new_windows = self.get_windows(motions)
        # print("new_window: ", new_windows.shape)

        self.data.append(new_windows)
        self.data = torch.cat(self.data)
        # print(type(self.data))


        train_len = self.data.shape[0] * 95 // 100
        self.test_set = self.data[train_len:, ...]
        self.data = self.data[:train_len, ...]
        self.data_reverse = torch.tensor(self.data.numpy()[..., ::-1].copy())

        self.reset_length_flag = 0
        self.virtual_length = 0
        print('Window count: {}, total frame: {}'.format(len(self), self.total_frame))

    def reset_length(self, length):
        self.reset_length_flag = 1
        self.virtual_length = length

    def __len__(self):
        if self.reset_length_flag:
            return self.virtual_length
        else:
            return self.data.shape[0]

    def __getitem__(self, item):
        if isinstance(item, int): item %= self.data.shape[0]
        if self.args.data_augment == 0 or np.random.randint(0, 2) == 0:
            return self.data[item]
        else:
            return self.data_reverse[item]

    def get_windows(self, motions):
        new_windows = []

        for motion in motions:
            if motion.shape[0] < self.args.window_size:
                add = self.args.window_size - motion.shape[0]
                temp = np.empty((add, motion.shape[1], motion.shape[2]))
                temp[:,:,:] = motion[-1,:,:]
                motion = np.vstack([motion,temp])
            # print(motion.shape)
            self.total_frame += motion.shape[0]
            self.motion_length.append(motion.shape[0])
            step_size = self.args.window_size // 2
            window_size = step_size * 2
            n_window = motion.shape[0] // step_size - 1
            # print('step_size: ', step_size)
            # print("window_size: ", window_size)
            # print("n_window: ", n_window)
            for i in range(n_window):
                begin = i * step_size
                end = begin + window_size

                new = motion[begin:end, :]
                new = new[np.newaxis, ...]

                new_window = torch.tensor(new, dtype=torch.float32)
                new_windows.append(new_window)

        return torch.cat(new_windows)
