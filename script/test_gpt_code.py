import sys
sys.path.append('./')

import sapien.core as sapien
from collections import OrderedDict
import pdb
from envs import *
import yaml
import importlib
import traceback

def class_decorator(task_name):
    envs_module = importlib.import_module(f'envs.{task_name}')
    importlib.reload(envs_module)
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No such task")
    return env_instance

def create_config_file(task_name, task_config_path):
    # if not os.path.isfile(task_config_path):
    if True:
        data = {
            'task_name': task_name,
            'render_freq': 0,
            'use_seed': False,
            'collect_data': True,
            'save_path': './data',
            'dual_arm': True,
            'st_episode': 0 ,
            'camera_w': 320,
            'camera_h': 240,
            'pcd_crop': True ,
            'pcd_down_sample_num': 1024,
            'episode_num': 100,
            'save_freq': 15,
            'head_camera_fovy': 45,
            'save_type':{
                'raw_data': False,
                'pkl': True
            },
            'data_type':{
                'rgb': True,
                'observer': False,
                'depth': True,
                'pointcloud': True,
                'conbine': False,
                'endpose': True,
                'qpos': True,
                'mesh_segmentation': False,
                'actor_segmentation': False,
            }
        }
        with open(task_config_path, 'w') as f:
            yaml.dump(data,f,default_flow_style = False,sort_keys=False)

def test_run(task_name):
    task = class_decorator(task_name)
    task_config_path = f'./task_config/{task_name}.yml'
    create_config_file(task_name, task_config_path)
    with open(task_config_path, 'r', encoding='utf-8') as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    return run(task, args)

def get_actor_keys_and_points_discription(task_name):
    task_config_path = f'./task_config/{task_name}.yml'
    create_config_file(task_name, task_config_path)
    with open(task_config_path, 'r', encoding='utf-8') as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    Demo_class = class_decorator(task_name)
    Demo_class.setup_demo(now_ep_num=0, seed = 0, **args)
    res_dic = Demo_class.get_actor_points_discription()
    
    return list(Demo_class.actor_name_dic.keys()), list(Demo_class.actor_data_dic.keys()), res_dic
    

def run(Demo_class, args):
    epid = 0 
    suc_num = 0    
    fail_num = 0

    error_list = ["The code can not run", "The left arm failed to grasp the object", 
                  "The right arm failed to grasp the object", "The target position of the object is incorrect."]
    
    error_num = [0, 0, 0, 0]
    
    test_num = 10
    while epid < test_num:
        error_id = None
        try:
            Demo_class.setup_demo(now_ep_num=suc_num, seed = epid, **args)
            Demo_class.play_once()

            if Demo_class.plan_success and Demo_class.check_success():
                suc_num+=1
            else:
                fail_num +=1
                if Demo_class.left_plan_success == False:
                    error_id = 1
                elif Demo_class.right_plan_success == False:
                    error_id = 2
                else:
                    error_id = 3
            Demo_class.close()
            if (args['render_freq']):
                Demo_class.viewer.close()
            epid +=1
        except Exception as e:
            error_id = 0
            stack_trace = traceback.format_exc()
            error_list[0] = str(stack_trace)
            fail_num +=1
            Demo_class.close()
            if (args['render_freq']):
                Demo_class.viewer.close()
            epid +=1
        # pdb.set_trace()
        if error_id is not None:
            error_num[error_id] += 1

    print(f'\nComplete test, success rate: {suc_num}/{test_num}')
    max_error_count = max(error_num)
    max_error_index = error_num.index(max_error_count)

    # return success rate, error message, error count
    return suc_num/test_num, error_list[max_error_index], max_error_count