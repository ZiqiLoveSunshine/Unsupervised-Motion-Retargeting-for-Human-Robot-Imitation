from models import create_model
import os
from data import create_dataset, get_character_names
import option_parser
from visualization.skeleton_plot import pyplot_skeleton
from visualization.draw_animation import draw_3Danimation,draw_3Danimation_2person

if __name__ == "__main__":
    save_dir = '../train_model/pretrained_less/pretrained_less_deep0817after'
    # save_dir = './pretrained_less_deep0811after'
    para_path = os.path.join(save_dir, 'para.txt')
    with open(para_path, 'r') as para_file:
        argv_ = para_file.readline().split()[1:]
        args = option_parser.get_parser().parse_args(argv_)
    args.cuda_device = 'cpu'
    args.is_train = False
    args.eval_seq = 2
    args.save_dir = save_dir
    character_names = get_character_names(args)
    print('character names: ', character_names)
    dataset = create_dataset(args, character_names)
    # print(args)
    model = create_model(args, dataset)
    model.load(epoch=args.epoch_num-1)
    # model.load(epoch=30000)

    motions = dataset[-3]
    model.set_input(motions)
    model.test()
    model.get_result()

    all_result = model.all_recons

    topo = list(dataset.joint_topology)
    topo[0] = 0

    num = 0
    print(len(all_result[num]))
    # B
    retarget_gt = all_result[num][0].cpu().clone().numpy()
    retarget = all_result[num][1].cpu().clone().numpy()

    
    # A
    res_gt = all_result[num][2].cpu().clone().numpy()
    res = all_result[num][3].cpu().clone().numpy()
    print(res.shape)
    i = 15
    # # reconstruction compare
    # res_gt_plot = pyplot_skeleton(topo, res_gt[i], show = False, color = "red", relative = False)
    # pyplot_skeleton(topo, res[i], ax = res_gt_plot, relative = False)

    # # retargeting compare
    # retarget_gt_plot = pyplot_skeleton(topo, retarget_gt[i], show = False, color = "red", relative = False)
    # pyplot_skeleton(topo, retarget[i], ax = retarget_gt_plot, color = "blue", relative = False)

    # # recon et retart
    # res_plot = pyplot_skeleton(topo, res[i], show = False, relative = False)
    # pyplot_skeleton(topo, retarget[i], ax = res_plot, color = "blue", relative = False)

    if "deep" in save_dir:
        atype = "deep"
    elif "cycle" in save_dir:
        atype = "cycle"
    # draw_3Danimation(topo, res_gt, "results/"+atype+"_res_gt.gif", world_position=True)
    # draw_3Danimation(topo, res, "results/"+atype+"_res.gif", world_position=True)
    # draw_3Danimation(topo, retarget, "results/"+atype+"_retarget.gif", world_position=True)

    # draw_3Danimation_2person(topo, res_gt, "red", res, "black", "results/"+atype+"_res_comparaison.gif", world_position = True)
    # draw_3Danimation_2person(topo, retarget_gt, "red", retarget, "blue", "results/"+atype+"_retarget_comparaison.gif", world_position = True)
    draw_3Danimation_2person(topo, res, "black", retarget, "blue", "results/"+atype+"_res_retarget_comparaison.gif", world_position = True)