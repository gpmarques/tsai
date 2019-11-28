# -*- coding: utf-8 -*-
# @Author: pedrotorres
# @Date:   2019-10-26 14:21:23
# @Last Modified by:   pedrotorres
# @Last Modified time: 2019-11-03 16:11:59

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


def read_points_2d(filename):
	points = list()
	with open(filename, 'r') as f:
		for line in f:
			line = line.split()
			points.append([float(line[0]), float(line[1])])

	return np.array(points)


def read_points_3d(filename):
	points = list()
	with open(filename, 'r') as f:
		for line in f:
			line = line.split()
			points.append([float(line[0]), float(line[1]), float(line[2])])

	return np.array(points)


def load_image(path):
	return cv2.imread(path)[:,:,::-1]


def draw_epipolar_lines(F, img_left, img_right, pts_left, pts_right):
	"""
	Draw the epipolar lines given the fundamental matrix, left right images
	and left right datapoints
	You do not need to modify anything in this function, although you can if
	you want to.
	:param F: 3 x 3; fundamental matrix
	:param img_left:
	:param img_right:
	:param pts_left: N x 2
	:param pts_right: N x 2
	:return:
	"""
	# lines in the RIGHT image
	# corner points
	p_ul = np.asarray([0, 0, 1])
	p_ur = np.asarray([img_right.shape[1], 0, 1])
	p_bl = np.asarray([0, img_right.shape[0], 1])
	p_br = np.asarray([img_right.shape[1], img_right.shape[0], 1])

	# left and right border lines
	l_l = np.cross(p_ul, p_bl)
	l_r = np.cross(p_ur, p_br)

	fig, ax = plt.subplots()
	ax.imshow(img_right)
	ax.autoscale(False)
	ax.scatter(pts_right[:, 0], pts_right[:, 1], marker='o', s=20, c='yellow',
		edgecolors='red')
	for p in pts_left:
		p = np.hstack((p, 1))[:, np.newaxis]
		l_e = np.dot(F, p).squeeze()  # epipolar line
		p_l = np.cross(l_e, l_l)
		p_r = np.cross(l_e, l_r)
		x = [p_l[0]/p_l[2], p_r[0]/p_r[2]]
		y = [p_l[1]/p_l[2], p_r[1]/p_r[2]]
		ax.plot(x, y, linewidth=1, c='blue')

	# lines in the LEFT image
	# corner points
	p_ul = np.asarray([0, 0, 1])
	p_ur = np.asarray([img_left.shape[1], 0, 1])
	p_bl = np.asarray([0, img_left.shape[0], 1])
	p_br = np.asarray([img_left.shape[1], img_left.shape[0], 1])

	# left and right border lines
	l_l = np.cross(p_ul, p_bl)
	l_r = np.cross(p_ur, p_br)

	fig, ax = plt.subplots()
	ax.imshow(img_left)
	ax.autoscale(False)
	ax.scatter(pts_left[:, 0], pts_left[:, 1], marker='o', s=20, c='yellow',
		edgecolors='red')

	for p in pts_right:
		p = np.hstack((p, 1))[:, np.newaxis]
		l_e = np.dot(F.T, p).squeeze()  # epipolar line
		p_l = np.cross(l_e, l_l)
		p_r = np.cross(l_e, l_r)
		x = [p_l[0]/p_l[2], p_r[0]/p_r[2]]
		y = [p_l[1]/p_l[2], p_r[1]/p_r[2]]
		ax.plot(x, y, linewidth=1, c='blue')

def visualize_points(points_2d, points_2d_projected):
    plt.scatter(points_2d[:, 0], points_2d[:, 1], marker='s', c='red', label='Real points')
    plt.scatter(points_2d_projected[:, 0], points_2d_projected[:, 1], marker='+', c='blue', label='Projected points')
    plt.legend()
    plt.show()

def project_points(M, points_3d):
    n = len(points_3d)
    points_3d = np.concatenate((np.array(points_3d), np.ones((n, 1))), axis=1)
    points_3d_projected = np.dot(M, points_3d.T).T
    u, v = points_3d_projected[:, 0] / points_3d_projected[:, 2], points_3d_projected[:, 1] / points_3d_projected[:, 2]
    points_2d = np.hstack((u[:, np.newaxis], v[:, np.newaxis]))

    return points_2d

def plot3dview(points_3d, camera_center):
    """
    Visualize the actual 3D points and the estimated 3D camera center.
    You do not need to modify anything in this function, although you can if you
    want to.
    :param points_3d: N x 3
    :param camera_center: 1 x 3
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='blue',
        marker='o', s=10, depthshade=0)
    camera_center = camera_center.squeeze()
    ax.scatter(camera_center[0],  camera_center[1], camera_center[2], c='red',
        marker='x', s=20, depthshade=0)

    # draw vertical lines connecting each point to ground
    min_z = min(points_3d[:, 2])

    for p in points_3d:
        x, y, z = p
        ax.plot3D(xs=[x, x], ys=[y, y], zs=[z, min_z], c='black', linewidth=1)

    x, y, z = camera_center
    ax.plot3D(xs=[x, x], ys=[y, y], zs=[z, min_z], c='black', linewidth=1)
