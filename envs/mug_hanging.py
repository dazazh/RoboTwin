
from .base_task import Base_task
from .utils import *
import numpy as np
import sapien

class mug_hanging(Base_task):
    def setup_demo(self,is_test=False,**kwags):
        super()._init(**kwags)
        self.create_table_and_wall()
        self.load_robot()
        self.setup_planner()
        self.load_camera(kwags.get('camera_w', 640),kwags.get('camera_h', 480))
        self.pre_move()
        if is_test:
            self.id_list = [1,3,4,6,7,8,9]
        else:
            self.id_list = [0,2]

        self.middle_pose_of_left_arm = [0.05, -0.15, 0.75]
        self.load_actors(f"./task_config/scene_info/{self.task_name[4:]}.json")
        self.step_lim = 800
    
    def pre_move(self):
        render_freq = self.render_freq
        self.render_freq=0
        self.together_open_gripper(save_freq=None)
        self.render_freq = render_freq

    def play_once(self):
        pass

    def check_success(self):
        mug = self.actor_name_dic['mug']
        mug_data = self.actor_data_dic['mug_data']
        rack = self.actor_name_dic['rack']
        mug_target_pose = self.get_actor_goal_pose(mug,mug_data)
        eps = np.array([0.05,0.03,0.02])
        return np.all(abs(mug_target_pose - rack.get_pose().p + [0.07,0.02,-0.035]) < eps) and self.is_right_gripper_open()
