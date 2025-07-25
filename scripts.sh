train.py --dis_linear true --save_dir ./pretrained_less_deep0814after --model deep --epoch_num 40001 --lambda_rec 10 --lambda_cycle 4 --lambda_gan 1

train.py --dis_linear true --save_dir ./pretrained_less_cycle0814after --model cycle --epoch_num 40001 --lambda_rec 10 --lambda_cycle 1 --lambda_gan 1