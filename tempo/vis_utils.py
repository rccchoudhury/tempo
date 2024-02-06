import pdb
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch
import cv2

COCO_SKELETON = [ (5, 7), (7, 9), (6, 8),
                 (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (5, 6),
                 (5, 11), (6, 12), (11, 12), (0, 6), (0, 5)]

H36M_SKELETON = [(0, 4), (4, 5), (5, 6), (0, 1), (1, 2), (2, 3), (0, 7),
                 (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14),
                 (14, 15), (15, 16)]

def draw_3d_skeleton(ax, joints, color):
    for link in COCO_SKELETON:
        #if 0 in joints_gt_single[link[0], :3] or 0 in joints_gt_single[link[1], :3]:
        #        continue
        link_indices = [_i for _i in link]
        xs_3d = joints[link_indices, 0]
        ys_3d = joints[link_indices, 1]
        zs_3d = joints[link_indices, 2]
        #ax.scatter(xs_3d, ys_3d, zs_3d, color=colors[gt_idx], s=25)
        ax.plot(xs_3d, ys_3d, zs_3d, color=color, alpha=min(1.0, joints[0, 4]))

def draw_2d_skeleton(image, pose, color):
    for j1, j2 in COCO_SKELETON:
        cv2.line(
            image, (int(pose[j1, 0]), int(pose[j1, 1])),
            (int(pose[j2, 0]), int(pose[j2, 1])),
            color=color,
            lineType=cv2.LINE_AA,
            thickness=5)
    return image
           

def imshow_multiview_keypoints_3d_simple(joints_pred=None,
                                         joints_gt=None,
                                         roots_gt=None,
                                         roots_pred=None,
                                         cameras=None,
                                         tracker_idxs=None,
                                         bbox3d=None):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xlim(-2500, 2000)
    ax.set_ylim(-3500, 3000)
    ax.set_zlim(0, 2000)

    colors = np.array([
        '#89cff0', '#f0aa89', 'blue', 'purple', 'orange', 'brown', 'olive', 'gray',
        'pink', 'cyan', 'black'
    ])

    if cameras is not None:
        for time_idx in range(len(cameras)):
            # start plotting cameras.
            assert isinstance(cameras, list)
            for idx, camera in enumerate(cameras[time_idx]):
                if 'extrinsics' in camera:
                    extrinsic = camera['extrinsics']
                else:
                    extrinsic = np.concatenate([
                        np.concatenate([camera['R'], camera['T']], axis=1),
                        np.array([[0, 0, 0, 1]])
                    ],
                                            axis=0)
                #fx, fy = camera['f'][0][0], camera['f'][1][0]
                #aspect_ratio = 0.3
                # This is a rendering shortcut. Will need to revert this
                # later in order to compute the FOVs
                fx, fy = 150, 150
                aspect_ratio = 1.0
                vertex_std = np.array(
                    [[0, 0, 0, 1], [fx * aspect_ratio, -fy * aspect_ratio, fx, 1],
                    [fx * aspect_ratio, fy * aspect_ratio, fx, 1],
                    [-fx * aspect_ratio, fy * aspect_ratio, fx, 1],
                    [-fx * aspect_ratio, -fy * aspect_ratio, fx, 1]])

                vertex_transformed = vertex_std @ extrinsic.T
                vertices = vertex_transformed[:, :3]
                ax.scatter(
                    vertices[:, 0],
                    vertices[:, 1],
                    vertices[:, 2],
                    color=colors[idx],
                    marker='o',
                    s=1)

                meshes = [[
                    vertex_transformed[0, :-1], vertex_transformed[1][:-1],
                    vertex_transformed[2, :-1]
                ],
                        [
                            vertex_transformed[0, :-1],
                            vertex_transformed[2, :-1],
                            vertex_transformed[3, :-1]
                        ],
                        [
                            vertex_transformed[0, :-1],
                            vertex_transformed[3, :-1],
                            vertex_transformed[4, :-1]
                        ],
                        [
                            vertex_transformed[0, :-1],
                            vertex_transformed[4, :-1],
                            vertex_transformed[1, :-1]
                        ],
                        [
                            vertex_transformed[1, :-1],
                            vertex_transformed[2, :-1],
                            vertex_transformed[3, :-1],
                            vertex_transformed[4, :-1]
                        ]]

                ax.add_collection3d(
                    Poly3DCollection(
                        meshes,
                        facecolors=None,
                        linewidths=0.3,
                        edgecolors='black',
                        alpha=0.35))


    # if joints_gt is not None:
    #     for time_idx in range(len(joints_gt)):
    #         for gt_idx in range(len(joints_gt[time_idx])):
    #             if gt_idx == 2: continue
    #             joints_gt_single = joints_gt[time_idx][gt_idx, :, :]
    #             for link in COCO_SKELETON:
    #                 if 0 in joints_gt_single[link[0], :3] or 0 in joints_gt_single[link[1], :3]:
    #                         continue
    #                 link_indices = [_i for _i in link]
    #                 xs_3d = joints_gt_single[link_indices, 0]
    #                 ys_3d = joints_gt_single[link_indices, 1]
    #                 zs_3d = joints_gt_single[link_indices, 2]
    #                 #ax.scatter(xs_3d, ys_3d, zs_3d, color=colors[gt_idx], s=25)
    #                 ax.plot(xs_3d, ys_3d, zs_3d, color=colors[gt_idx], alpha=0.7**(time_idx) * (0 if gt_idx == 2 else 1))
    
    if joints_pred is not None:
        for person_idx, joints_gt_single in enumerate(joints_pred):
            if joints_gt_single[0, -1] < 0.3: continue
            for link in COCO_SKELETON:
                #if 0 in joints_gt_single[link[0], :3] or 0 in joints_gt_single[link[1], :3]:
                #        continue
                link_indices = [_i for _i in link]
                xs_3d = joints_gt_single[link_indices, 0]
                ys_3d = joints_gt_single[link_indices, 1]
                zs_3d = joints_gt_single[link_indices, 2]
                #ax.scatter(xs_3d, ys_3d, zs_3d, color=colors[gt_idx], s=25)
                ax.plot(xs_3d, ys_3d, zs_3d, color=colors[int(joints_gt_single[0, 3])],
                            alpha=min(1.0, joints_gt_single[0, 4]))

    #fig.tight_layout()
    fig.canvas.draw()
    img_w, img_h = fig.canvas.get_width_height()
    img_vis = np.frombuffer(
        fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(img_h, img_w, -1)

    img_vis = torch.from_numpy(img_vis.copy()).permute(2, 0, 1)
    plt.close(fig)

    return img_vis