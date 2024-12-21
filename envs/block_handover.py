
from .base_task import Base_task
from .utils import *
import sapien
import math

class block_handover(Base_task):
    def setup_demo(self,**kwags):
        super()._init(**kwags)
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

    def create_target_block_data(self, half_size):
        contact_discription_list = []
        test_matrix = np.array([[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,0,1]])
        test_matrix[:3,:3] = t3d.euler.euler2mat(0,0,np.pi) @ test_matrix[:3,:3]
        contact_points_list = []

        data = {
            'center': [0,0,0],
            'extents': half_size,
            'scale': [1,1,1],                                     # 缩放
            'target_pose': [[[1,0,0,0],[0,1,0,0],[0,0,1,half_size[2]],[0,0,0,1]]],              # 目标点矩阵
            'contact_points_pose' : contact_points_list,    # 抓取点矩阵（多个）
            'transform_matrix': np.eye(4).tolist(),           # 模型到标轴的旋转矩阵
            "functional_matrix": [],         # 功能点矩阵
            'contact_points_discription': contact_discription_list,    # 抓取点描述
            'contact_points_group': [],
            'contact_points_mask': [],
            'target_point_discription': ["The center point on the bottom of the box."]
        }

        return data

    def create_grasp_block_data(self, half_size):
        contact_discription_list = []
        
        contact_points_list = [
                [[0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0.05], [0, 0, 0, 1]], # front
                [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0.05], [0, 0, 0, 1]], # left
                [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0.05], [0, 0, 0, 1]], # right
                [[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0.05], [0, 0, 0, 1]], # back

                [[0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, -0.05], [0, 0, 0, 1]], # front
                [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, -0.05], [0, 0, 0, 1]], # left
                [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, -0.05], [0, 0, 0, 1]], # right
                [[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, -0.05], [0, 0, 0, 1]], # back
            ]

        functional_matrix = np.eye(4)
        functional_matrix[:3,:3] = t3d.euler.euler2mat(np.pi,0,0)
        functional_matrix[:3,3] = np.array([0,0,-half_size[2]])
        data = {
            'center': [0,0,0],
            'extents': half_size,
            'scale': [1,1,1],                                     # scale
            'target_pose': [[[1,0,0,0],[0,1,0,0],[0,0,1,-half_size[2]],[0,0,0,1]]],              # traget points matrix
            'contact_points_pose' : contact_points_list,    # contact points matrix list
            'transform_matrix': np.eye(4).tolist(),           # transform matrix
            "functional_matrix": [functional_matrix.tolist()],         # functional points matrix
            'contact_points_discription': contact_discription_list,    # contact points discription
            'contact_points_group': [[0, 1, 2, 3], [4, 5, 6, 7]],
            'contact_points_mask': [True, True],
            'target_point_discription': ["The center point on the bottom of the block."]
        }

        return data

    def play_once(self):
        pass

    def check_success(self):
        grasp_block = self.actor_name_dic['grasp_block']
        target_block = self.actor_name_dic['target_block']
        box_pos = grasp_block.get_pose().p
        target_pose = target_block.get_pose().p
        if box_pos[2] < 0.78:
            self.actor_pose = False
        eps = 0.02
        right_endpose = self.get_right_endpose_pose()
        endpose_target_pose = [0.241,-0.129,0.889,0,-0.7,-0.71,0]
        return abs(box_pos[0] - target_pose[0]) < eps and abs(box_pos[1] - target_pose[1]) < eps and abs(box_pos[2] - 0.85) < 0.0015  and self.is_right_gripper_open()