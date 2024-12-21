
from .base_task import Base_task
from .utils import *
import sapien

class empty_cup_place(Base_task):
    def setup_demo(self,**kwags):
        super()._init(**kwags)
        self.create_table_and_wall()
        self.load_robot()
        self.setup_planner()
        self.load_camera(kwags.get('camera_w', 640),kwags.get('camera_h', 480))
        self.pre_move()
        self.load_actors(f"./task_config/scene_info/{self.task_name[4:]}.json")
        self.step_lim = 500
    
    def pre_move(self):
        render_freq = self.render_freq
        self.render_freq=0
        self.together_open_gripper(save_freq=None)
        self.render_freq = render_freq

    def play_once(self):
        pass
    
    def check_success(self):
        eps = 0.025
        coaster = self.actor_name_dic['coaster']
        cup = self.actor_name_dic['cup']
        coaster_pose = coaster.get_pose().p
        cup_pose = cup.get_pose().p
        return abs(cup_pose[0] - coaster_pose[0])<eps  and  abs(cup_pose[1] - coaster_pose[1])<eps and (cup_pose[2] - 0.792) < 0.005
