import sys
from torch.utils.data.dataloader import DataLoader
from data import create_dataset, get_character_names
import option_parser
import os
from option_parser import try_mkdir
import time
from models import create_model


def main():
    args = option_parser.get_args()
    # parse = option_parser.get_parser()
    # print(parse.get_value())
    characters = get_character_names(args)
    log_path = os.path.join(args.save_dir, 'logs/')
    try_mkdir(args.save_dir)
    try_mkdir(log_path)
    with open(os.path.join(args.save_dir, 'para.txt'), 'w') as para_file:
        para_file.write(' '.join(sys.argv))

    dataset = create_dataset(args, characters)
    data_loader = DataLoader(dataset, batch_size = args.batch_size, shuffle=True, num_workers=0)

    model = create_model(args, dataset)
    # print(args.lambda_rec)
    if args.epoch_begin:
        model.load(epoch = args.epoch_begin, download=False)

    model.setup()

    start_time = time.time()

    for epoch in range(args.epoch_begin, args.epoch_num):
        for step, motions in enumerate(data_loader):
            for i, motion in enumerate(motions):
                print("epoch: {}, step: {}, i: {}".format(epoch,step,i))
                model.set_input(motion)
            model.optimize_parameters()

            if args.verbose:
                res = model.verbose()
                print('[{}/{}]\t[{}/{}]\t'.format(epoch, args.epoch_num, step, len(data_loader)), res)

        if epoch % 1000 == 0 or epoch == args.epoch_num - 1:
            model.save()

        model.epoch()

    end_tiem = time.time()
    print('training time', end_tiem - start_time)


if __name__ == '__main__':
    main()
