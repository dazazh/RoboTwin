
from .base_task import Base_task
from .utils import *
import sapien

class diverse_bottles_pick(Base_task):
    def setup_demo(self,is_test = False,**kwags):
        super()._init(**kwags)
        self.create_table_and_wall()
        self.load_robot()
        self.setup_planner()
        self.load_camera(kwags.get('camera_w', 640),kwags.get('camera_h', 480))
        self.pre_move()
        if is_test:
            self.id_list = [2*i + 1 for i in range(11)]
        else:
            self.id_list = [2*i for i in range(11)]
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
        red_target = [-0.06,-0.105]
        green_target = [0.06,-0.105]
        eps = 0.03
        bottle1 = self.actor_name_dic['bottle1']
        bottle2 = self.actor_name_dic['bottle2']
        bottle1_pose = bottle1.get_pose().p
        bottle2_pose = bottle2.get_pose().p
        if bottle1_pose[2] < 0.78 or bottle2_pose[2] < 0.78:
            self.actor_pose = False
        return abs(bottle1_pose[0]-red_target[0])<eps and abs(bottle1_pose[1]-red_target[1])<eps and bottle1_pose[2]>0.89 and\
               abs(bottle2_pose[0]-green_target[0])<eps and abs(bottle2_pose[1]-green_target[1])<eps and bottle2_pose[2]>0.89
