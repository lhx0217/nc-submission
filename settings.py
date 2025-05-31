import json
import math
import numpy as np


def arrange_points_in_square(n):
    # 计算正方形的边长
    side_length = math.ceil(math.sqrt(n))
    if side_length * (side_length - 1) >= n:
        rows = side_length
        cols = side_length - 1
    else:
        rows = side_length
        cols = side_length
    # 生成每个点的坐标
    points = []
    for row in range(rows):
        for col in range(cols):
            points.append(((col - cols / 2) * 2, (row - rows / 2) * 2))
    # 返回点的坐标
    return points


import random
import math

def generate_points_3d(n, region_size=60, max_attempts=1000):
    points = []
    random.seed(2020)  # 固定随机种子，便于复现

    # 生成第一个点
    first_point = (0, 0, 0)
    points.append(first_point)

    while len(points) < n:
        valid = False
        candidate = None

        for _ in range(max_attempts):
            # 在立方体中随机生成候选点
            candidate = (
                random.uniform(-region_size / 2, region_size / 2),
                random.uniform(-region_size / 2, region_size / 2),
                random.uniform(-region_size / 2, region_size / 2)
            )

            # 计算与已有点的最小距离
            min_distance = float('inf')
            for p in points:
                dx = p[0] - candidate[0]
                dy = p[1] - candidate[1]
                dz = p[2] - candidate[2]
                distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                if distance < min_distance:
                    min_distance = distance

                # 如果距离太近就跳出
                if distance < 2:
                    break
            else:
                # 距离在合理范围内则接受
                if 3 <= min_distance <= 8:
                    valid = True
                    break

        if valid:
            points.append(candidate)
        else:
            raise RuntimeError(f"无法在 {max_attempts} 次尝试内生成第 {len(points) + 1} 个点")

    return points


def generate_points(n, region_size=50, max_attempts=1000):
    points = []
    region_size = 60
    random.seed(3)  # 任意整数作为种子
    # 生成第一个随机点
    first_point = (0, 0)
    points.append(first_point)

    # 生成后续点
    while len(points) < n:
        candidate = None
        valid = False

        # 尝试生成候选点
        for _ in range(max_attempts):
            # 生成随机候选点
            candidate = (
                random.uniform(-region_size / 2, region_size / 2),
                random.uniform(-region_size / 2, region_size / 2)
            )

            # 计算与所有已有点的最小距离
            min_distance = float('inf')
            for p in points:
                dx = p[0] - candidate[0]
                dy = p[1] - candidate[1]
                distance = math.sqrt(dx ** 2 + dy ** 2)
                if distance < min_distance:
                    min_distance = distance

                # 检查最小距离是否≥2
                if distance < 2:
                    break
            else:
                # 检查最近邻是否在[2,5]区间
                if 3 <= min_distance <= 8:
                    valid = True
                    break

        if valid:
            points.append(candidate)
        else:
            raise RuntimeError(f"无法在{max_attempts}次尝试内生成第{len(points) + 1}个点")

    return points

def gen_settings(center, num):
    """
    生成num个无人机的settings文件, 参差矩形初始状态
    """
    initial_content = {'SeeDocsAt': 'https://github.com/Microsoft/AirSim/blob/master/docs/settings.md',
                       'SettingsVersion': 1.2,
                       'SimMode': 'Multirotor',
                       'ViewMode': 'Manual',
                       # "ClockSpeed": 1,
                       "ClockSpeed": 1,  # 设置慢动作的速度，小于1的值会减慢仿真时间
                       # "Recording": {
                       #     "RecordInterval": 0.05,
                       #     "Cameras": [
                       #         {"CameraName": "0", "ImageType": 0, "PixelsAsFloat": False, "Compress": True,
                       #          "VehicleName": "UAV0", "Folder": "D:/coding/Aceberg_Pro-main/control/data"}
                       #     ]
                       # },
                       # "CameraDefaults": {
                       #     "CaptureSettings": [
                       #         {
                       #             "ImageType": 0,
                       #             "Width": 1024,
                                   "Height": 960,
                       #             "FOV_Degrees": 90
                       #         }
                       #     ]
                       # },
                       'Vehicles': None
                       }
    with open(r"/home/lhx/Documents/AirSim/settings.json", 'w') as f:
        if num == 0:
            json.dump(initial_content, f)
            return
        uav_dir = {}
        points = generate_points(num)[:num] + np.array(center)
        for n in range(num):
            uav_dir[f'UAV{n}'] = {'VehicleType': 'SimpleFlight', 'X': points[n][0], 'Y': points[n][1], 'Z': 0, 'Yaw': 0,
                                  "DefaultVehicleState": "Inactive",
                                  # "Sensors": {
                                  #         "LidarSensor0": {
                                  #             "SensorType": 6,
                                  #             "Range": 15,  # 扫描距离，单位为米
                                  #             "Enabled": True,
                                  #             "NumberOfChannels": 8,
                                  #             "RotationsPerSecond": 10,
                                  #             "PointsPerSecond": 1000,
                                  #             "X": 0, "Y": 0, "Z": -1,
                                  #             "Roll": 0, "Pitch": 0, "Yaw": 0,
                                  #             "VerticalFOVUpper": -90,
                                  #             "VerticalFOVLower": -80,
                                  #             "DrawDebugPoints": True,
                                  #             "DataFrame": "SensorLocalFrame"
                                  #         },
                                  #     "LidarSensor": {
                                  #         "SensorType": 6,
                                  #         "Range": 8,  # 扫描距离，单位为米
                                  #         "Enabled": True,
                                  #         "NumberOfChannels": 8,
                                  #         "RotationsPerSecond": 10,
                                  #         "PointsPerSecond": 2000,
                                  #         "X": 0, "Y": 0, "Z": -1,
                                  #         "Roll": 0, "Pitch": 0, "Yaw": 0,
                                  #         "VerticalFOVUpper": 35,
                                  #         "VerticalFOVLower": -35,
                                  #         "DrawDebugPoints": False,
                                  #         "DataFrame": "SensorLocalFrame"
                                  #     }
                                  # }
                                  }
        initial_content['Vehicles'] = uav_dir
        json.dump(initial_content, f)
    nested_list = []
    for coord in points:
        nested_list.append(list(coord) + [0])
    return np.array(nested_list)


def gen_random_2d(num, center, length):
    '''
    输入点数量, 矩形中心坐标, 矩形边长，输出在二维平面矩形内的随机点, 第三维为0
    '''
    # 计算矩形的半边长
    half_length = length / 2

    # 生成随机点的x和y坐标，它们位于以center为中心的矩形内
    x = np.random.uniform(center[0] - half_length, center[0] + half_length, num)
    y = np.random.uniform(center[1] - half_length, center[1] + half_length, num)

    # 将x和y坐标组合成二维数组，并添加第三维（z）坐标，其值为0
    points = np.column_stack((x, y, np.zeros(num)))

    return points


def gen_random_3d(num, center, length):
    '''
    输入点数量, 立方体中心坐标, 立方体边长，输出在三维矩形内的随机点
    '''
    # 计算立方体的半边长
    half_length = length / 2

    # 生成随机点的x, y, z坐标，它们位于以center为中心的立方体内
    x = np.random.uniform(center[0] - half_length, center[0] + half_length, num)
    y = np.random.uniform(center[1] - half_length, center[1] + half_length, num)
    z = np.random.uniform(center[2] - half_length, center[2] + half_length, num)

    # 将x, y, z坐标组合成三维数组
    points = np.column_stack((x, y, z))

    return points


if __name__ == "__main__":
    print(gen_settings(np.array([0, 0]), 50))
