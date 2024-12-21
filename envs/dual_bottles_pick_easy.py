
from .base_task import Base_task
from .utils import *
import sapien

class dual_bottles_pick_easy(Base_task):
    def setup_demo(self,**kwags):
        super()._init(**kwags)
        self.create_table_and_wall()
        self.load_robot()
        self.setup_planner()
        self.load_camera(kwags.get('camera_w', 640),kwags.get('camera_h', 480))
        self.pre_move()
        self.left_bottle_target_position = [-0.06,-0.105, 0.92]
        self.right_bottle_target_position = [0.06,-0.105, 0.92]
        self.load_actors(f"./task_config/scene_info/{self.task_name[4:]}.json")
        self.step_lim = 400
    
    def pre_move(self):
        render_freq = self.render_freq
        self.render_freq=0

        self.together_close_gripper(save_freq=None)
        self.together_open_gripper(save_freq=None)

        self.render_freq = render_freq

    def play_once(self):
        pass

    def check_success(self):
        red_target = [-0.055,-0.105]
        green_target = [0.055,-0.105]
        eps = 0.03
        red_bottle = self.actor_name_dic['red_bottle']
        green_bottle = self.actor_name_dic['green_bottle']
        red_bottle_pose = red_bottle.get_pose().p
        green_bottle_pose = green_bottle.get_pose().p
        if red_bottle_pose[2] < 0.78 or green_bottle_pose[2] < 0.78:
            self.actor_pose = False
        return abs(red_bottle_pose[0]-red_target[0])<eps and abs(red_bottle_pose[1]-red_target[1])<eps and red_bottle_pose[2]>0.9 and \
               abs(green_bottle_pose[0]-green_target[0])<eps and abs(green_bottle_pose[1]-green_target[1])<eps and green_bottle_pose[2]>0.9
