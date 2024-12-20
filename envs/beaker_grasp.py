from .base_task import Base_task
from .utils import *
import sapien

class beaker_grasp(Base_task):
    def setup_demo(self, **kwargs):
        super()._init(**kwargs, table_static=True)
        self.create_table_and_wall()
        self.load_robot()
        self.setup_planner()
        self.load_camera(kwargs.get('camera_w', 640), kwargs.get('camera_h', 480))
        self.pre_move()
        self.load_actors(f"./task_config/scene_info/{self.task_name[4:]}.json")

    def pre_move(self):
        render_freq = self.render_freq
        self.render_freq = 0
        self.together_open_gripper(save_freq=None)
        self.render_freq = render_freq
        
    def play_once(self):
        pass
    
    # Check success
    def check_success(self):
        beaker = self.actor_name_dic['beaker']
        beaker_pose_p = np.array(beaker.get_pose().p)
        beaker_pose_q = np.array(beaker.get_pose().q)

        if beaker_pose_q[0] < 0:
            beaker_pose_q *= -1

        eps = 0.025
        coaster = self.actor_name_dic['coaster']
        coaster_pose = coaster.get_pose().p
        return abs(beaker_pose_p[0] - coaster_pose[0])<eps  and  abs(beaker_pose_p[1] - coaster_pose[1])<eps and (beaker_pose_p[2] - 0.792) < 0.005
        # target_pose_p = np.array([0,-0.08])
        # target_pose_q = np.array([0.5,0.5,-0.5,-0.5])
        # eps = np.array([0.05,0.02,0.05,0.05,0.05,0.05])
        # return np.all(abs(beaker_pose_p[:2] - target_pose_p) < eps[:2]) and np.all(abs(beaker_pose_q - target_pose_q) < eps[-4:] ) and self.is_left_gripper_open() and self.is_right_gripper_open()