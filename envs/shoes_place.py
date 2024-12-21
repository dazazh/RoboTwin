
from .base_task import Base_task
from .utils import *
import math
import sapien

class shoes_place(Base_task):
    def setup_demo(self,is_test = False, **kwags):
        super()._init(**kwags)
        self.create_table_and_wall()
        self.load_robot()
        self.setup_planner()
        self.load_camera(kwags.get('camera_w', 640),kwags.get('camera_h', 480))
        self.pre_move()
        if is_test:
            self.id_list = [2*i+1 for i in range(5)]
        else:
            self.id_list = [2*i for i in range(5)]
        self.load_actors(f"./task_config/scene_info/{self.task_name[4:]}.json")
        self.step_lim = 600
    
    def pre_move(self):
        render_freq = self.render_freq
        self.render_freq=0
        self.together_open_gripper(save_freq=None)
        self.render_freq = render_freq


    def create_block_data(self, half_size):
        contact_discription_list = []
        test_matrix = np.array([[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,0,1]])
        test_matrix[:3,:3] = t3d.euler.euler2mat(0,0,np.pi) @ test_matrix[:3,:3]
        # print(test_matrix.tolist())
        contact_points_list = []

        data = {
            'center': [0,0,0],
            'extents': half_size,
            'scale': [1,1,1],                                     # 缩放
            'target_pose': [[[1,0,0,0],[0,1,0,-0.06],[0,0,1,half_size[2]],[0,0,0,1]], [[1,0,0,0],[0,1,0,0.06],[0,0,1,half_size[2]],[0,0,0,1]]],              # 目标点矩阵
            'contact_points_pose' : contact_points_list,    # 抓取点矩阵（多个）
            'transform_matrix': np.eye(4).tolist(),           # 模型到标轴的旋转矩阵
            "functional_matrix": [[0., 1., 0., 0.], [0., 0., -1., 0.], [1., 0., 0., 0.], [0., 0., 0., 1.]],         # 功能点矩阵
            'contact_points_discription': contact_discription_list,    # 抓取点描述
            'contact_points_group': [],
            'contact_points_mask': [],
            'target_point_discription': ["The center point on the top of the box." ]
        }

        return data

    def get_target_grap_pose(self,shoe_rpy):
        if math.fmod(math.fmod(shoe_rpy[2] + shoe_rpy[0], 2 * math.pi) + 2 * math.pi, 2*math.pi) < math.pi:
            grasp_matrix = np.array([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
            target_quat = [-0.707,0,-0.707,0]
        else:
            grasp_matrix = np.eye(4)
            target_quat = [0,0.707,0,-0.707]
        return grasp_matrix, target_quat

    def play_once(self):
        pass

    def check_success(self):
        left_shoe = self.actor_name_dic['left_shoe']
        right_shoe = self.actor_name_dic['right_shoe']
        left_shoe_pose_p = np.array(left_shoe.get_pose().p)
        left_shoe_pose_q = np.array(left_shoe.get_pose().q)
        right_shoe_pose_p = np.array(right_shoe.get_pose().p)
        right_shoe_pose_q = np.array(right_shoe.get_pose().q)
        if left_shoe_pose_q[0] < 0:
            left_shoe_pose_q *= -1
        if right_shoe_pose_q[0] < 0:
            right_shoe_pose_q *= -1
        target_pose_p = np.array([0,-0.13])
        target_pose_q = np.array([0.5,0.5,-0.5,-0.5])
        eps = np.array([0.02,0.02,0.05,0.05,0.05,0.05])
        return np.all(abs(left_shoe_pose_p[:2] - (target_pose_p - [0,0.06])) < eps[:2]) and np.all(abs(left_shoe_pose_q - target_pose_q) < eps[-4:]) and \
               np.all(abs(right_shoe_pose_p[:2] - (target_pose_p + [0,0.06])) < eps[:2]) and np.all(abs(right_shoe_pose_q - target_pose_q) < eps[-4:]) and self.is_left_gripper_open() and self.is_right_gripper_open()
        
