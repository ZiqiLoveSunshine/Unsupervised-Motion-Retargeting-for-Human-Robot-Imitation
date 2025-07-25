from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from visualization.skeleton_plot import set_axes_equal



def draw_3Danimation(topology, p1_pos, save_path, world_position = True):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    
    def init(topology, p1_offsets):
        p1_lines = []
        p1_total_offsets = np.zeros_like(p1_offsets)
        for j, i in enumerate(topology):
    #         print("i: {}, j: {}".format(i, j))
            if world_position:
                p1_total_offsets[j] = p1_offsets[j]
            else:
                p1_total_offsets[j] = p1_offsets[j] + p1_total_offsets[i]
            p1_stick_line = ax.plot(p1_total_offsets[[i,j], 0], p1_total_offsets[[i,j], 2], 
                p1_total_offsets[[i,j], 1])[0]
            p1_stick_line._verts3d = p1_total_offsets[[i,j], 0], p1_total_offsets[[i,j], 2], p1_total_offsets[[i,j], 1]
            p1_lines.append(p1_stick_line)
        return p1_lines

    def draw_animation_skeleton(i, topology, p1_all_offsets, p1_lines):
        # Plot a skeleton in 3d
        p1_offsets = p1_all_offsets[i]
        p1_total_offsets = np.zeros_like(p1_offsets)
    #     print("total offset: ", p1_total_offsets.shape)
        for j, i in enumerate(topology):
    #         print("i: {}, j: {}".format(i, j))
            if world_position:
                p1_total_offsets[j] = p1_offsets[j]
            else:
                p1_total_offsets[j] = p1_offsets[j] + p1_total_offsets[i]
            p1_lines[j]._verts3d = p1_total_offsets[[i,j], 0], p1_total_offsets[[i,j], 2], p1_total_offsets[[i,j], 1]
        set_axes_equal(ax)
        return p1_lines
    
#     poppy_topology = dataset.joint_topologies[0]
    p1_lines = init(topology, p1_pos[0])
    total_len = p1_pos.shape[0]
    # poppy_lines
    anim = animation.FuncAnimation(fig = fig, func = draw_animation_skeleton, 
                                   frames = total_len, fargs = (topology, p1_pos, p1_lines), blit = True)
    anim.save(save_path)



def draw_3Danimation_2person(topology, p1_pos, c1, p2_pos, c2, save_path, world_position = True):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    
    def init(topology, p1_offsets, p2_offsets, c1, c2):
        p1_lines = []
        p2_lines = []
        p1_total_offsets = np.zeros_like(p1_offsets)
        p2_total_offsets = np.zeros_like(p2_offsets)
        for j, i in enumerate(topology):
    #         print("i: {}, j: {}".format(i, j))
            if world_position:
                p1_total_offsets[j] = p1_offsets[j]
                p2_total_offsets[j] = p2_offsets[j]
            else:
                p1_total_offsets[j] = p1_offsets[j] + p1_total_offsets[i]
                p2_total_offsets[j] = p2_offsets[j] + p2_total_offsets[i]
            p1_stick_line = ax.plot(p1_total_offsets[[i,j], 0], p1_total_offsets[[i,j], 2], 
                                    p1_total_offsets[[i,j], 1], color = c1)[0]
            p1_stick_line._verts3d = p1_total_offsets[[i,j], 0], p1_total_offsets[[i,j], 2], p1_total_offsets[[i,j], 1]
            p1_lines.append(p1_stick_line)

            p2_stick_line = ax.plot(p2_total_offsets[[i,j], 0], p2_total_offsets[[i,j], 2], 
                                    p2_total_offsets[[i,j], 1], color = c2)[0]
            p2_stick_line._verts3d = p2_total_offsets[[i,j], 0], p2_total_offsets[[i,j], 2], p2_total_offsets[[i,j], 1]
            p2_lines.append(p2_stick_line)
        return p1_lines, p2_lines

    def draw_animation_skeleton(i, topology, p1_all_offsets, p1_lines, p2_all_offsets, p2_lines):
        # Plot a skeleton in 3d
        p1_offsets = p1_all_offsets[i]
        p2_offsets = p2_all_offsets[i]
        p1_total_offsets = np.zeros_like(p1_offsets)
        p2_total_offsets = np.zeros_like(p2_offsets)
    #     print("total offset: ", p1_total_offsets.shape)
        for j, i in enumerate(topology):
    #         print("i: {}, j: {}".format(i, j))
            if world_position:
                p1_total_offsets[j] = p1_offsets[j]
                p2_total_offsets[j] = p2_offsets[j]
            else:
                p1_total_offsets[j] = p1_offsets[j] + p1_total_offsets[i]
                p2_total_offsets[j] = p2_offsets[j] + p2_total_offsets[i]
            p1_lines[j]._verts3d = p1_total_offsets[[i,j], 0], p1_total_offsets[[i,j], 2], p1_total_offsets[[i,j], 1]
            p2_lines[j]._verts3d = p2_total_offsets[[i,j], 0], p2_total_offsets[[i,j], 2], p2_total_offsets[[i,j], 1]
        set_axes_equal(ax)
        return p1_lines+p2_lines
    
#     poppy_topology = dataset.joint_topologies[0]
    p1_lines,p2_lines = init(topology, p1_pos[0], p2_pos[0],c1,c2)
    total_len = min(p1_pos.shape[0],p2_pos.shape[0])
    # poppy_lines
    anim = animation.FuncAnimation(fig = fig, func = draw_animation_skeleton, 
                                   frames = total_len, fargs = (topology, p1_pos, p1_lines, p2_pos, p2_lines), blit = True)
    anim.save(save_path)