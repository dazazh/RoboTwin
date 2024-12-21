
from .base_task import Base_task
from .utils import *
import sapien
import math

class blocks_stack_hard(Base_task):

    def setup_demo(self,**kwags):
        super()._init(**kwags)
        self.create_table_and_wall()
        self.load_robot()
        self.setup_planner()
        self.load_camera(kwags.get('camera_w', 640),kwags.get('camera_h', 480))
        self.pre_move()
        self.block1_target_pose = [0, -0.1, 0.75]
        self.load_actors(f"./task_config/scene_info/{self.task_name[4:]}.json")
        self.step_lim = 850

    def pre_move(self):
        render_freq = self.render_freq
        self.render_freq=0
        self.together_open_gripper(save_freq=None)
        self.render_freq = render_freq

    def create_block_data(self, half_size):
        contact_discription_list = []
        contact_points_list = [
                [[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], # top_down(front)
                [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], # top_down(right)
                [[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], # top_down(left)
                [[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], # top_down(back)
            ]
        functional_matrix = np.eye(4)
        functional_matrix[:3,:3] = t3d.euler.euler2mat(np.pi,0,0)
        functional_matrix[:3,3] = np.array([0,0,-half_size[2]])

        data = {
            'center': [0,0,0],
            'extents': half_size,
            'scale': [1,1,1],                                     # scale
            'target_pose': [[[1,0,0,0],[0,1,0,0],[0,0,1,half_size[2]],[0,0,0,1]]],              # target points matrix
            'contact_points_pose' : contact_points_list,    # contact points matrix list
            'transform_matrix': np.eye(4).tolist(),           # transform matrix
            "functional_matrix": [functional_matrix.tolist()],         # functional points matrix
            'contact_points_discription': contact_discription_list,    # contact points discription
            'contact_points_group': [[0, 1, 2, 3]],
            'contact_points_mask': [True],
            'target_point_discription': ["The top surface center of the block." ],
            'functional_point_discription': ["Point0: The center point on the bottom of the block, and functional axis is vertical bottom side down"]
        }

        return data
    
    def play_once(self):
        pass
        
    def check_success(self):
        block1 = self.actor_name_dic['block1']
        block2 = self.actor_name_dic['block2']
        block3 = self.actor_name_dic['block3']
        block1_pose = block1.get_pose().p
        block2_pose = block2.get_pose().p
        block3_pose = block3.get_pose().p
        target_pose = [0,-0.1]
        eps = [0.03,0.03,0.01]
        return np.all(abs(block1_pose - np.array(target_pose + [0.765])) < eps) and \
               np.all(abs(block2_pose - np.array(target_pose + [0.815])) < eps) and \
               np.all(abs(block3_pose - np.array(target_pose + [0.865])) < eps) and self.is_left_gripper_open() and self.is_right_gripper_open()