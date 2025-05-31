import os

import airsim
import numpy as np


def scale_coordinates(coordinates, scale_factor):
    """将所有点扩大scale_factor倍"""
    # 创建一个空列表来存储新的坐标
    scaled_coordinates = []

    # 遍历每一个坐标点
    for coord in coordinates:
        # 计算新的坐标点，将每一个坐标值乘以扩大因子
        new_coord = [x * scale_factor for x in coord]
        # 将新的坐标点添加到列表中
        scaled_coordinates.append(new_coord)

    return np.array(scaled_coordinates)


def matlab(points):
    # 将浮点坐标转换为整数
    points_int = np.round(points).astype(int)
    ne = []
    for points in points_int:
        pre = points
        print(f'Gray_image({pre[0] + 15},{pre[1] + 15},{pre[2] + 15})=0;')
        ne.append(pre)


def file_write(kmeans, record_rate, record_pos, record_comm, record_time):
    t = 'tree3'
    if kmeans:
        with open(f'./data/run_data/rate_km_{t}.json', 'w+') as f:
            f.write(str(record_rate))
        with open(f'./data/run_data/pos_km_{t}.json', 'w+') as f:
            f.write(str(record_pos))
        with open(f'./data/run_data/comm_km_{t}.json', 'w+') as f:
            f.write(str(record_comm))
        with open(f'./data/run_data/time_km_{t}.json', 'w+') as f:
            f.write(str(record_time))
    else:
        with open(f'./data/run_data/rate_ms_{t}.json', 'w+') as f:
            f.write(str(record_rate))
        with open(f'./data/run_data/pos_ms_{t}.json', 'w+') as f:
            f.write(str(record_pos))
        with open(f'./data/run_data/comm_ms_{t}.json', 'w+') as f:
            f.write(str(record_comm))
        with open(f'./data/run_data/time_ms_{t}.json', 'w+') as f:
            f.write(str(record_time))


tools = [
    {
        "type": "function",  # 约定的字段 type，目前支持 function 作为值
        "function": {  # 当 type 为 function 时，使用 function 字段定义具体的函数内容
            "name": "generate_points_on_sphere",  # 函数的名称，请使用英文大小写字母、数据加上减号和下划线作为函数名称
            "description": """ 
				The function generates uniformly distributed points on the surface of a sphere.\n
				graph = generate_points_on_sphere(4, 3) return a numpy array.
			""",  # 函数的介绍，在这里写上函数的具体作用以及使用场景，以便 Kimi 大模型能正确地选择使用哪些函数
            "parameters": {  # 使用 parameters 字段来定义函数接收的参数
                "type": "object",  # 固定使用 type: object 来使 Kimi 大模型生成一个 JSON Object 参数
                "required": ["radius", "height"],  # 使用 required 字段告诉 Kimi 大模型哪些参数是必填项
                "properties": {  # properties 中是具体的参数定义，你可以定义多个参数
                    "radius": {  # 在这里，key 是参数名称，value 是参数的具体定义
                        "type": "number",  # 使用 type 定义参数类型
                        "description": """
							The radius of the sphere. Recommend 4.
						"""  # 使用 description 描述参数以便 Kimi 大模型更好地生成参数
                    },
                    "height": {  # 在这里，key 是参数名称，value 是参数的具体定义
                        "type": "number",  # 使用 type 定义参数类型
                        "description": """
							The height of the center of the sphere. Recommend 3.
						"""  # 使用 description 描述参数以便 Kimi 大模型更好地生成参数
                    }
                }
            }
        }
    },
    {
        "type": "function",  # 约定的字段 type，目前支持 function 作为值
        "function": {  # 当 type 为 function 时，使用 function 字段定义具体的函数内容
            "name": "generate_dna",  # 函数的名称，请使用英文大小写字母、数据加上减号和下划线作为函数名称
            "description": """ 
                This function generates a set of points in the shape of DNA. return a numpy array.\n
            """,  # 函数的介绍，在这里写上函数的具体作用以及使用场景，以便 Kimi 大模型能正确地选择使用哪些函数
        }
    },
    {
        "type": "function",  # 约定的字段 type，目前支持 function 作为值
        "function": {  # 当 type 为 function 时，使用 function 字段定义具体的函数内容
            "name": "generate_points_on_cylinder",  # 函数的名称，请使用英文大小写字母、数据加上减号和下划线作为函数名称
            "description": """ 
				The function Generate uniformly distributed points on the curved surface of a cylinder, without distributing on the top and bottom two circular faces.\n
				sample: graph = generate_points_on_cylinder({'radius': 4, 'height': 10, 'center_height': 3}) return a numpy array.
			""",  # 函数的介绍，在这里写上函数的具体作用以及使用场景，以便 Kimi 大模型能正确地选择使用哪些函数
            "parameters": {  # 使用 parameters 字段来定义函数接收的参数
                "type": "object",  # 固定使用 type: object 来使 Kimi 大模型生成一个 JSON Object 参数
                "required": ["radius", "height", "center_height"],  # 使用 required 字段告诉 Kimi 大模型哪些参数是必填项
                "properties": {  # properties 中是具体的参数定义，你可以定义多个参数
                    "radius": {  # 在这里，key 是参数名称，value 是参数的具体定义
                        "type": "number",  # 使用 type 定义参数类型
                        "description": """
							The radius of the cylinder. Recommend 4.
						"""  # 使用 description 描述参数以便 Kimi 大模型更好地生成参数
                    },
                    "height": {  # 在这里，key 是参数名称，value 是参数的具体定义
                        "type": "number",  # 使用 type 定义参数类型
                        "description": """
							The height of the cylinder. Recommend 10.
						"""  # 使用 description 描述参数以便 Kimi 大模型更好地生成参数
                    },
                    "center_height": {  # 在这里，key 是参数名称，value 是参数的具体定义
                        "type": "integer",  # 使用 type 定义参数类型
                        "description": """
							The height of the center of the cylinder's base circle. Recommend 3.
						"""  # 使用 description 描述参数以便 Kimi 大模型更好地生成参数
                    }
                }
            }
        }
    },
    {
        "type": "function",  # 约定的字段 type，目前支持 function 作为值
        "function": {  # 当 type 为 function 时，使用 function 字段定义具体的函数内容
            "name": "generate_square_points",  # 函数的名称，请使用英文大小写字母、数据加上减号和下划线作为函数名称
            "description": """ 
				The function Generate a set of uniformly distributed rectangular points on a plane.\n
				sample: graph = generate_square_points({'length': 20, 'width': 20, 'height': 3}) return a numpy array.
				generate_square_points output a list.
			""",  # 函数的介绍，在这里写上函数的具体作用以及使用场景，以便 Kimi 大模型能正确地选择使用哪些函数
            "parameters": {  # 使用 parameters 字段来定义函数接收的参数
                "type": "object",  # 固定使用 type: object 来使 Kimi 大模型生成一个 JSON Object 参数
                "required": ["length", "width", "height"],  # 使用 required 字段告诉 Kimi 大模型哪些参数是必填项
                "properties": {  # properties 中是具体的参数定义，你可以定义多个参数
                    "length": {  # 在这里，key 是参数名称，value 是参数的具体定义
                        "type": "number",  # 使用 type 定义参数类型
                        "description": """
							The length of the rectangle. it should be less than 2 * drone_number. length recommend 20.
						"""  # 使用 description 描述参数以便 Kimi 大模型更好地生成参数
                    },
                    "width": {  # 在这里，key 是参数名称，value 是参数的具体定义
                        "type": "number",  # 使用 type 定义参数类型
                        "description": """
							The width of the rectangle. it should be less than 2 * drone_number. width recommend 20.
						"""  # 使用 description 描述参数以便 Kimi 大模型更好地生成参数
                    },
                    "height": {  # 在这里，key 是参数名称，value 是参数的具体定义
                        "type": "number",  # 使用 type 定义参数类型
                        "description": """
							The height of the surface of the square. height recommend 3.
						"""  # 使用 description 描述参数以便 Kimi 大模型更好地生成参数
                    }
                }
            }
        }
    },
    {
        "type": "function",  # 约定的字段 type，目前支持 function 作为值
        "function": {  # 当 type 为 function 时，使用 function 字段定义具体的函数内容
            "name": "generate_circle_points",  # 函数的名称，请使用英文大小写字母、数据加上减号和下划线作为函数名称
            "description": """ 
				The function Generate a set of uniformly distributed circle points on a plane.\n
				sample: graph = generate_square_points({'radius': 5}) return a numpy array.
				generate_circle_points output a numpy array.
			""",  # 函数的介绍，在这里写上函数的具体作用以及使用场景，以便 Kimi 大模型能正确地选择使用哪些函数
            "parameters": {  # 使用 parameters 字段来定义函数接收的参数
                "type": "object",  # 固定使用 type: object 来使 Kimi 大模型生成一个 JSON Object 参数
                "required": ["radius"],  # 使用 required 字段告诉 Kimi 大模型哪些参数是必填项
                "properties": {  # properties 中是具体的参数定义，你可以定义多个参数
                    "radius": {  # 在这里，key 是参数名称，value 是参数的具体定义
                        "type": "number",  # 使用 type 定义参数类型
                        "description": """
							The radius of the circle. Recommend 5.
						"""  # 使用 description 描述参数以便 Kimi 大模型更好地生成参数
                    }
                }
            }
        }
    },
    {
        "type": "function",  # 约定的字段 type，目前支持 function 作为值
        "function": {  # 当 type 为 function 时，使用 function 字段定义具体的函数内容
            "name": "drone_swarm_formation",  # 函数的名称，请使用英文大小写字母、数据加上减号和下划线作为函数名称
            "description": """ 
                usage: drone_swarm_formation(graph)\n return nothing
				This function takes in a set of scattered points, driving the drone swarm to self-organize into a shape composed of the scattered points.
				the algorithm we use corresponds to one drone being responsible for multiple target points, we have already done this in function 
				'drone_swarm_formation'.\n 
			""",  # 消除了kimi对目标点数等于无人机数的误解
            "parameters": {  # 使用 parameters 字段来定义函数接收的参数
                "type": "object",  # 固定使用 type: object 来使 Kimi 大模型生成一个 JSON Object 参数
                "required": ["graph"],  # 使用 required 字段告诉 Kimi 大模型哪些参数是必填项
                "properties": {  # properties 中是具体的参数定义，你可以定义多个参数
                    "graph": {  # 在这里，key 是参数名称，value 是参数的具体定义
                        "type": "array",  # 使用 type 定义参数类型
                        "description": """
							This array should first be generated by a function tool starting with "generate_", and then input into this function.
						"""  # 使用 description 描述参数以便 Kimi 大模型更好地生成参数
                    }
                }
            }
        }
    },
    {
        "type": "function",  # 约定的字段 type，目前支持 function 作为值
        "function": {  # 当 type 为 function 时，使用 function 字段定义具体的函数内容
            "name": "drone_swarm_move_to_position",  # 函数的名称，请使用英文大小写字母、数据加上减号和下划线作为函数名称
            "description": """ 
                usage: drone_swarm_move_to_position(destination), return nothing
                !!!this function can be called only after drone_swarm_formation, because the swarm needs a formation first!!!
				This function takes a three-dimensional velocity vector, causing the swarm to move to desired position.
                The positive and negative values of the first dimension represent forward and backward, respectively. 
                The positive and negative values of the second dimension represent left and right, respectively. 
                The positive and negative values of the third dimension represent up and down, respectively.\n
			""",  # 消除了kimi对目标点数等于无人机数的误解
            "parameters": {  # 使用 parameters 字段来定义函数接收的参数
                "type": "object",  # 固定使用 type: object 来使 Kimi 大模型生成一个 JSON Object 参数
                "required": ["destination"],  # 使用 required 字段告诉 Kimi 大模型哪些参数是必填项
                "properties": {  # properties 中是具体的参数定义，你可以定义多个参数
                    "destination": {  # 在这里，key 是参数名称，value 是参数的具体定义
                        "type": "array",  # 使用 type 定义参数类型
                        "description": """
							This array should be a three dimensional vector. 
						"""  # 使用 description 描述参数以便 Kimi 大模型更好地生成参数
                    }
                }
            }
        }
    },
    {
        "type": "function",  # 约定的字段 type，目前支持 function 作为值
        "function": {  # 当 type 为 function 时，使用 function 字段定义具体的函数内容
            "name": "drone_swarm_stop",  # 函数的名称，请使用英文大小写字母、数据加上减号和下划线作为函数名称
            "description": """ 
                !!!usage: drone_swarm_stop().
                stops all current actions.
			""",  # 消除了kimi对目标点数等于无人机数的误解
        }
    },
    {
        "type": "function",  # 约定的字段 type，目前支持 function 作为值
        "function": {  # 当 type 为 function 时，使用 function 字段定义具体的函数内容
            "name": "get_swarm_center",  # 函数的名称，请使用英文大小写字母、数据加上减号和下划线作为函数名称
            "description": """ 
                !!!usage: get_swarm_center().
                this function returns the center of swarm in numpy array, such as array([0,0,0]).
			""",  # 消除了kimi对目标点数等于无人机数的误解
        }
    },
]
