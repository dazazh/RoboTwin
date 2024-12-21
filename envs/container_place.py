
from .base_task import Base_task
from .utils import *
import sapien

class container_place(Base_task):
    def setup_demo(self,is_test = False,**kwags):
        super()._init(**kwags)
        self.create_table_and_wall()
        self.load_robot()
        self.setup_planner()
        self.load_camera(kwags.get('camera_w', 640),kwags.get('camera_h', 480))
        self.pre_move()
        if is_test:
            self.id_list = [0,1,2,3,4,6,7]
        else:
            self.id_list = [8,9]
        self.load_actors(f"./task_config/scene_info/{self.task_name[4:]}.json")
        self.step_lim = 350
    
    def pre_move(self):
        render_freq = self.render_freq
        self.render_freq=0
        self.together_open_gripper(save_freq=None)
        self.render_freq = render_freq

    def play_once(self):
        pass

    def check_success(self):
        container = self.actor_name_dic['container']
        container_data = self.actor_data_dic['container_data']
        container_pose = self.get_actor_goal_pose(container,container_data)
        target_pose = np.array([0,-0.05, 0.74])
        eps = np.array([0.02,0.02, 0.01])
        left_gripper = self.active_joints[34].get_drive_target()[0]
        right_gripper = self.active_joints[36].get_drive_target()[0]
        endpose_z = max(self.get_right_endpose_pose().p[2], self.get_left_endpose_pose().p[2])
        return np.all(abs(container_pose - target_pose) < eps) and left_gripper > 0.04 and right_gripper > 0.04 and endpose_z > 0.98 and self.is_left_gripper_open() and self.is_right_gripper_open()