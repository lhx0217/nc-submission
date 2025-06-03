import open3d as o3d
import numpy as np

# 1. 读取 STL 网格
mesh = o3d.io.read_triangle_mesh("flower.stl")

# 检查是否读取成功
if not mesh.has_triangles():
    raise ValueError("该 STL 文件不包含三角面片")

# 2. 法线计算（某些操作需要）
mesh.compute_vertex_normals()

# 3. 从面片中均匀采样点（例如采样 100,000 个点）
pcd = mesh.sample_points_poisson_disk(number_of_points=8000)


# 4. 获取原始点坐标
points = np.asarray(pcd.points)

# 5. 计算几何中心
center = np.mean(points, axis=0)

# 6. 居中点云
points_centered = points - center
pcd.points = o3d.utility.Vector3dVector(points_centered)
points = np.array(pcd.points)
# 可选：上色
pcd.paint_uniform_color([0.3, 0.6, 1.0])

# 7. 可视化
o3d.visualization.draw_geometries([pcd], window_name="Dense Centered Point Cloud")
