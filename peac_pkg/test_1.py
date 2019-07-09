# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 10:25:02 2019

Creates a sample data set and identifies the peaks of the 
data from interpolated contours. Demonstrates the PEAC method's ability to 
find peaks spatially in a 2D space for a nth factor.

@author: ecramer

"""
#import sys, os
#os.chdir('F:\peac\peac_pkg')
#sys.path.append('F:\peac\peac_pkg')

import numpy as np
import matplotlib.pyplot as plt
from contourer import Contourer
from mpl_toolkits import mplot3d

def func(x, y):
    return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2

grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]
grid_z = func(grid_x, grid_y)

# plot the grid to show the contours of the underlying function
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(grid_x, grid_y, grid_z, 50, cmap='binary')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Contours of Sample Data')
plt.tight_layout()

# calculate values at random points to interpolate
# this simulates the recorded data from an experiment
np.random.seed(42)
points = np.random.rand(1000, 2)
values = func(points[:,0], points[:,1])

# plot the sampled data over the underlying function as a scatter
fig = plt.figure()
plt.imshow(func(grid_x, grid_y).T, extent=(0,1,0,1), origin='lower')
plt.plot(points[:,0], points[:,1], 'k.', ms=1)
plt.title('Sample Data')
plt.show()
plt.close()

# calculate the contours from the incomplete data
# adjust the distance between peaks to capture all of the high points
cc = Contourer()
cc.fit(points, values, **{'_peak_distance':2})

# show the calculated contours and the peaks over the same space
x = np.arange(0, cc._resolution+1, 1)
x = np.interp(x, (x.min(), x.max()), (cc.X_[:, 0].min(), cc.X_[:, 0].max()))
y = np.arange(0, cc._resolution+1, 1)
y = np.interp(y, (y.min(), y.max()), (cc.X_[:, 1].min(), cc.X_[:, 1].max()))   
xx, yy = np.meshgrid(x, y)
plot_arr = np.asarray([[x[a[0]] for a in cc.peak_coors_[0]], [y[a[1]] for a in cc.peak_coors_[0]]]).T

im = plt.imshow(cc.contours_[0], extent=(0,1,0,1), origin='lower', cmap="cividis")
plt.scatter(cc.transformed_peaks_[0][:, 1], cc.transformed_peaks_[0][:, 0], c = 'red', marker='x')
plt.title('Peaks Found')
plt.colorbar(im)
plt.show()
plt.close()