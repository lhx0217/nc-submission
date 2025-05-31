import copy

from openai import OpenAI
import re
import argparse
from utils import tools
from graph_base import *
from main import *
import math
import numpy as np
import threading
import os
import json
import time
from colorama import init, Fore, Style

graph = None


def exec_func(code):
    global graph
    exec(code)


parser = argparse.ArgumentParser()
parser.add_argument("--sysprompt", type=str, default="system_prompts/airsim_basic.txt")
args = parser.parse_args()

print("Initializing ChatGPT...")
client = OpenAI(
    api_key="sk-Snmt7yNILYDEowH3znPKzVKgsIQ3Uvr0wfbg26J9A6zd62Gj",
    base_url="https://api.moonshot.cn/v1",
)

with open(args.sysprompt, "r") as f:
    sysprompt = f.read()

chat_history = [
    {
        "role": "system",
        "content": sysprompt
    },
]


def ask(prompt, add_history=1):
    if add_history:
        chat_history.append(
            {
                "role": "user",
                "content": prompt,
            }
        )
        completion = client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=chat_history,
            temperature=0,
            tools=tools
        )
        chat_history.append(
            {
                "role": "assistant",
                "content": completion.choices[0].message.content,
            }
        )
    else:
        chat_history.append(
            {
                "role": "system",
                "content": prompt,
            }
        )
        completion = client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=chat_history,
            temperature=0,
            tools=tools
        )
    # print("==========================")
    # print(completion.choices[0])
    # print("==========================")
    return completion.choices[0].message.content


print(f"Done.")

code_block_regex = re.compile(r"```(.*?)```", re.DOTALL)


def extract_python_code(content):
    code_blocks = code_block_regex.findall(content)
    if code_blocks:
        full_code = "\n".join(code_blocks)
        if full_code.startswith("python"):
            full_code = full_code[7:]
        code = full_code.split('\n')
        return code
    else:
        return None


class colors:  # You may need to change color settings
    RED = "\033[31m"
    ENDC = "\033[m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"


print(Fore.YELLOW + "Welcome to the AirSim chatbot! I am ready to help you with your AirSim questions and commands.")
change = False  # 更改当前代码

while True:
    if not change:
        user_question = input(Fore.BLUE + "User> " + colors.ENDC)
        if user_question == "!quit" or user_question == "!exit":
            break
        if user_question == "!clear":
            os.system("cls")
            continue
        response = ask(user_question)
        print(Fore.YELLOW + f"chatgpt> {response}")
    # response = '''```python
    # import numpy as np
    #
    # # Generate a set of points on a square plane
    # graph = generate_square_points({'length': 40, 'width': 40, 'height': 3})
    #
    # # Organize the drones into the square shape
    # drone_swarm_formation(graph)
    #
    # # Calculate the velocity vector to move the swarm to (20, 20)
    # velocity_vector = np.array([20, 20, 0]) - np.array(get_swarm_center({}))
    #
    # # Normalize the velocity vector to ensure the speed is less than 2m/s
    # velocity_magnitude = np.linalg.norm(velocity_vector)
    # if velocity_magnitude > 0:
    #     velocity_vector = velocity_vector / velocity_magnitude * 2
    #
    # # Move the swarm to the desired location
    # drone_swarm_speed({'speed': velocity_vector.tolist()})
    # ```'''
    else:
        change = False  # 改回change为否
    codes = extract_python_code(response)
    if codes is not None:
        codes.append('drone_swarm_stop() # stop all operation current running, and re-init swarm.')
        current_code = 'None'
        if change is True:
            break
        print(Fore.YELLOW + "Please wait while I run the code in AirSim...")
        next_action = ''
        mul_code = ''
        action_time = 'None'
        for code in codes:
            wait = True
            next_action += code
            if code == '':
                next_action = ''
            elif code[0] == '#':
                next_action = ''
            elif 'drone_swarm' not in code:
                next_action = ''
                mul_code += code
                try:
                    mul_code = mul_code.replace("functions.", "")
                    exec(mul_code)
                    print(Fore.CYAN + f"system execute: {mul_code}")
                    mul_code = ''
                except Exception as e:
                    print(Fore.RED + 'not end or run err: ', code, e)
            else:
                action_execution_status = 'Normal'
                while wait:
                    status = sim.record_rate
                    graph_center = copy.deepcopy(sim.graph_center)
                    destination = copy.deepcopy(sim.destination)
                    graph_center[-1] = -graph_center[-1]
                    destination[-1] = -destination[-1]
                    question = f"user's aim: {user_question}\n" \
                               f"current action: {current_code}\n" \
                               f"Current action executed time: {action_time}\n" \
                               f"Current action Execution Status: {action_execution_status}\n" \
                               f"your next action: {next_action}\n" \
                               f"enter_rate: {status[-1][0]}\ncover_rate: {status[-1][1]}\nuniformity: {status[-1][2]}\n" \
                               f"swarm_position: {graph_center} \n" \
                               f"swarm_destination: {destination} \n" \
                               f"destination_arrival_status: {destination_arrival_status} \n" \
                               f"Please tell me your choice, \`wait for current action\`, \`begin next action\` or \`change next action\`."
                    print(Fore.CYAN + 'System>\n' + question)
                    response = ask('System>\n' + question, add_history=0)
                    print(Fore.YELLOW + 'chatgpt> ', response)
                    if 'begin next action' in response:
                        code = code.replace("functions.", "")
                        try:
                            exec(code)
                            current_code = copy.deepcopy(code)
                            print(Fore.CYAN + f"system execute: {code}")
                            action_execution_status = 'Normal'
                        except Exception as err:
                            print(Fore.RED + 'err: ', err)
                            print(Fore.RED + f"execute_fail: {code}")
                            action_execution_status = 'Action Fail! ' + err
                        action_time = 0
                        wait = False
                        next_action = ''
                    elif 'wait for current action' in response:
                        action_time += 10
                        if action_time > 30 and np.linalg.norm(destination - graph_center) < 0.1:
                            action_execution_status = 'Errors may occur, after 30 seconds the swarm still cannot reach ' \
                                                      'the formation indicate the graph shape is not suitable for the swarm size.'
                    elif 'change next action' in response:
                        action_execution_status = 'Normal'
                        change = True
                        wait = False
                    time.sleep(10)
