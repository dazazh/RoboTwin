
from .base_task import Base_task
from .utils import *
import math
import sapien

class pick_apple_messy(Base_task):
    def setup_demo(self,is_test = False, **kwags):
        super()._init(**kwags)
        self.create_table_and_wall()
        self.load_robot()
        self.setup_planner()
        self.load_camera(kwags.get('camera_w', 640),kwags.get('camera_h', 480))
        self.pre_move()
        self.load_actors(f"./task_config/scene_info/{self.task_name[4:]}.json")
        # self.load_actors()
        self.step_lim = 250
    
    def pre_move(self):
        render_freq = self.render_freq
        self.render_freq=0
        self.together_open_gripper(save_freq=None)
        self.render_freq = render_freq

    def play_once(self):
        pass

    def check_success(self):
        apple = self.actor_name_dic['apple']
        apple_pose = apple.get_pose().p
        return apple_pose[2] > 0.81
