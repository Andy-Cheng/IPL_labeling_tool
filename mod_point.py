
import os
import numpy as np
import open3d as o3d



path_lidar = '/mnt/nas_kradar/kradar_dataset/dir_all/5/radar/cart_cacfar_15_5_150_150/00471_old.pcd'
points = np.asarray(o3d.io.read_point_cloud(path_lidar).points)
print(points.shape)



mask = (points[:, 2] <= 8) & (points[:, 2] >= 0)
filtered_points = points[mask]

#  4.33 2.22 1.90 
x, y, z = 61.58, 15.65, -0.06
range_delete = 5

mask2 =  (filtered_points[:, 2] <= z+range_delete) & (filtered_points[:, 2] >= z-range_delete) & (filtered_points[:, 1] <= y+range_delete) & (filtered_points[:, 1] >= y-range_delete)\
    & (filtered_points[:, 0] <= x+range_delete) & (filtered_points[:, 0] >= x-range_delete)
mask2 = ~mask2
filtered_points = filtered_points[mask2]


num_points = filtered_points.shape[0]
indices = np.random.choice(num_points, size=int(num_points * 0.5), replace=False)
selected_points = filtered_points[indices]
print(selected_points.shape)

# Create a new point cloud with the selected points
new_point_cloud = o3d.geometry.PointCloud()
new_point_cloud.points = o3d.utility.Vector3dVector(selected_points)

# Save the new point cloud to a PCD file
o3d.io.write_point_cloud("/mnt/nas_kradar/kradar_dataset/dir_all/5/radar/cart_cacfar_15_5_150_150/00471.pcd", new_point_cloud)