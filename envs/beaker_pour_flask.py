from .base_task import Base_task
from .utils import *
import sapien

class beaker_pour_flask(Base_task):
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
        beaker_data = self.actor_data_dic['beaker_data']
        conical_flask = self.actor_name_dic['conical_flask']
        conical_flask_data = self.actor_data_dic['conical_flask_data']

        beaker_pose = self.get_actor_functional_pose(beaker, beaker_data, id=0)  
        conical_flask_pose = self.get_actor_functional_pose(conical_flask, conical_flask_data, id=0)
        eps = np.array([0.05,0.03,0.1])
        return abs(beaker_pose[0] - conical_flask_pose[0]) < eps[0] and abs(beaker_pose[1] - conical_flask_pose[1]) < eps[1] and abs(beaker_pose[2] - conical_flask_pose[2]) < eps[2]