
from .base_task import Base_task
from .utils import *
import sapien

class dual_bottles_pick_easy(Base_task):
    def setup_demo(self,**kwags):
        super()._init(**kwags)
        self.create_table_and_wall()
        self.load_robot()
        self.setup_planner()
        self.load_camera()
        self.pre_move()
        self.load_actors()
        self.step_lim = 400
    
    def pre_move(self):
        render_freq = self.render_freq
        self.render_freq=0

        self.together_close_gripper(save_freq=None)
        self.together_open_gripper(save_freq=None)

        self.render_freq = render_freq

    def load_actors(self):
        # super().setup_scene()
        self.beaker,_ = rand_create_glb(
            self.scene,
            xlim=[[0.15,0.3],[-0.3,-0.15]],
            ylim=[[0.13,0.15]],
            zlim=[[0.8]],
            modelname="043_beaker",
            rotate_rand=False,
            qpos=[0.707,0.707,0,0],
            scale=(0.09,0.12,0.09),
            model_id=13
        )

        self.coaster,_ = rand_create_glb(
            self.scene,
            xlim=[[-0.05,0.1],[-0.1, 0.05]],
            ylim=[[0.13,0.15]],
            zlim=[[0.76]],
            modelname="019_coaster",
            rotate_rand=False,
            qpos=[0.707,0.707,0,0],
            scale=(0.09,0.12,0.09),
        )

        self.beaker.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.01
        self.coaster.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.01

        render_freq = self.render_freq
        self.render_freq = 0
        for _ in range(4):
            self.together_open_gripper(save_freq=None)
        self.render_freq = render_freq
    def check_success(self):
        beaker = self.beaker
        beaker_pose_p = np.array(beaker.get_pose().p)
        beaker_pose_q = np.array(beaker.get_pose().q)

        if beaker_pose_q[0] < 0:
            beaker_pose_q *= -1

        eps = 0.025
        coaster = self.coaster
        coaster_pose = coaster.get_pose().p
        return abs(beaker_pose_p[0] - coaster_pose[0])<eps  and  abs(beaker_pose_p[1] - coaster_pose[1])<eps and (beaker_pose_p[2] - 0.792) < 0.005