
from .base_task import Base_task
from .utils import *
import sapien

class block_hammer_beat(Base_task):

    def setup_demo(self,**kwags):
        super()._init(**kwags)
        self.create_table_and_wall()
        self.load_robot()
        self.setup_planner()
        self.load_camera(kwags.get('camera_w', 640),kwags.get('camera_h', 480))
        self.pre_move()
        self.load_actors(f"./task_config/scene_info/{self.task_name[4:]}.json")
        self.step_lim = 400

    def create_block_data(self, half_size):
        contact_discription_list = []
        test_matrix = np.array([[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,0,1]])
        test_matrix[:3,:3] = t3d.euler.euler2mat(0,0,np.pi) @ test_matrix[:3,:3]
        contact_points_list = [
                [[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], # top_down(front)
                [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], # top_down(right)
                [[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], # top_down(left)
                [[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], # top_down(back)
            ]

        data = {
            'center': [0,0,0],
            'extents': half_size,
            'scale': [1,1,1],                                     # scale
            'target_pose': [[[1,0,0,0],[0,1,0,0],[0,0,1,half_size[2]],[0,0,0,1]]],              # traget points matrix
            'contact_points_pose' : contact_points_list,    # contact points matrix list
            'transform_matrix': np.eye(4).tolist(),           # transform matrix
            "functional_matrix": [[[1,0,0,0],[0,1,0,0],[0,0,1,half_size[2]],[0,0,0,1]]],         # functional points matrix
            'contact_points_discription': contact_discription_list,    # contact points discription
            'contact_points_group': [[0, 1, 2, 3]],
            'contact_points_mask': [True],
            'target_point_discription': ["The center point on the bottom of the box."]
        }

        return data
    
    def pre_move(self):
        render_freq = self.render_freq
        self.render_freq=0
        self.together_open_gripper(save_freq=None)
        self.render_freq = render_freq

    def play_once(self):
        pass
        
    def check_success(self):
        hammer = self.actor_name_dic['hammer']
        hammer_data = self.actor_data_dic['hammer_data']
        block = self.actor_name_dic['block']
        hammer_target_pose = self.get_actor_functional_pose(hammer,hammer_data)[:3]
        block_pose = block.get_pose().p
        eps = np.array([0.02,0.02])
        return np.all(abs(hammer_target_pose[:2] - block_pose[:2])<eps) and hammer_target_pose[2] < 0.81 and hammer_target_pose[2] > 0.78