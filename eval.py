import os
from models import create_model
from data import create_dataset, get_character_names
import option_parser
import torch
from tqdm import tqdm


def eval(eval_seq, save_dir, test_device='cpu'):
    para_path = os.path.join(save_dir, 'para.txt')
    with open(para_path, 'r') as para_file:
        argv_ = para_file.readline().split()[1:]
        args = option_parser.get_parser().parse_args(argv_)
    args.cuda_device = test_device if torch.cuda.is_available() else 'cpu'
    args.is_train = False
    args.eval_seq = eval_seq
    args.save_dir = save_dir
    print(args)
    character_names = get_character_names(args)

    dataset = create_dataset(args, character_names)
    # print(args)
    model = create_model(args, dataset)
    model.load(epoch=args.epoch_num-1)

    for i, motions in tqdm(enumerate(dataset), total=len(dataset)):
        model.set_input(motions)
        model.test()
    final = torch.tensor(model.test_final)
    print("value final: ", final.mean())

if __name__ == '__main__':
    parser = option_parser.get_parser()
    args = parser.parse_args()
    args.save_dir = '../train_model/pretrained_less/pretrained_less_deep0815morning'
    eval(2, args.save_dir, args.cuda_device)
