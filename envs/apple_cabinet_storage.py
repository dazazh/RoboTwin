from .base_task import Base_task
from .utils import *
import sapien


class apple_cabinet_storage(Base_task):
    def setup_demo(self,**kwags):
        super()._init(**kwags,table_static=False)
        self.create_table_and_wall()
        self.load_robot()
        self.setup_planner()
        self.load_camera(kwags.get('camera_w', 640),kwags.get('camera_h', 480))
        self.pre_move()
        self.load_actors(f"./task_config/scene_info/{self.task_name[4:]}.json")
        self.step_lim = 600

    def pre_move(self):
        render_freq = self.render_freq
        self.render_freq=0
        self.together_open_gripper(save_freq=None)
        self.render_freq = render_freq
        
    def play_once(self):
        pass
        
    def check_success(self):
        cabinet = self.actor_name_dic['cabinet']
        apple = self.actor_name_dic['apple']
        cabinet_pos = cabinet.get_pose().p
        eps = 0.03
        apple_pose = apple.get_pose().p
        left_endpose = self.get_left_endpose_pose()
        target_pose = (cabinet_pos + np.array([-0.05,-0.27,-0.09])).tolist() + [0.5, -0.5, -0.5, 0.5]
        eps1 = 0.03
        return np.abs(apple_pose[2] - 0.797) < 0.015 and \
               np.all(abs(np.array(left_endpose.p.tolist() + left_endpose.q.tolist()) - target_pose) < eps1)