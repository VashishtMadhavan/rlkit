"""
Plotting script for AntMaze to see where exploration is taking us
"""
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import math
import gym

# TODO: add ability to parse data file from replay buffer
def parse_args():
	return None

def get_structure_plot(structure, size_scaling):
	# duplicate cells to plot the maze
	structure_plot = np.zeros(((len(structure) - 1) * 2, (len(structure[0]) - 1) * 2))
	for i in range(len(structure)):
		for j in range(len(structure[0])):
			cell = structure[i][j]
			if type(cell) is not int:
				cell = 0.3 if cell == 'r' else 0.7
			if i == 0:
				if j == 0:
					structure_plot[i, j] = cell
				elif j == len(structure[0]) - 1:
					structure_plot[i, 2 * j - 1] = cell
				else:
					structure_plot[i, 2 * j - 1:2 * j + 1] = cell
			elif i == len(structure) - 1:
				if j == 0:
					structure_plot[2 * i - 1, j] = cell
				elif j == len(structure[0]) - 1:
					structure_plot[2 * i - 1, 2 * j - 1] = cell
				else:
					structure_plot[2 * i - 1, 2 * j - 1:2 * j + 1] = cell
			else:
				if j == 0:
					structure_plot[2 * i - 1:2 * i + 1, j] = cell
				elif j == len(structure[0]) - 1:
					structure_plot[2 * i - 1:2 * i + 1, 2 * j - 1] = cell
				else:
					structure_plot[2 * i - 1:2 * i + 1, 2 * j - 1:2 * j + 1] = cell
	return structure_plot

def transform_xy(env, point):
	size_scaling = env.unwrapped.MAZE_SIZE_SCALING
	ori = env.unwrapped.get_ori()
	o_xy = np.array(env.unwrapped._find_robot())
	o_ij = (o_xy / size_scaling).astype(int)
	o_xy_plot = o_xy / size_scaling * 2
	robot_xy_plot = o_xy_plot + point / size_scaling * 2
	return robot_xy_plot

def get_random_xy(env, test_eps=10):
	goals = []
	for _ in range(test_eps):
		done = False; obs = env.reset()
		while not done:
			obs, rew, done, info = env.step(env.action_space.sample())
			goals.append(transform_xy(env, obs['achieved_goal']))
	return np.array(goals)

def plot_maze(args):
	plot_env = gym.make("AntMaze-v2")
	structure = plot_env.unwrapped.MAZE_STRUCTURE
	size_scaling = plot_env.unwrapped.MAZE_SIZE_SCALING

	# plot maze structure
	struct_plot = get_structure_plot(structure, size_scaling)
	fig, ax = plt.subplots()
	im = ax.pcolor(-np.array(struct_plot), cmap='gray', edgecolor='black', lw=0.05)
	x_labels = list(range(len(structure[0])))
	y_labels = list(range(len(structure)))

	ax.xaxis.set(ticks=2 * np.arange(len(x_labels)), ticklabels=x_labels)
	ax.yaxis.set(ticks=2 * np.arange(len(y_labels)), ticklabels=y_labels)

	# plot x,y coordinates agent
	pos = get_random_xy(plot_env, test_eps=1)
	plt.scatter(pos[:,0], pos[:,1], s=20, c='g', marker='x')
	plt.show()


if __name__ == "__main__":
	args = parse_args()
	plot_maze(args)