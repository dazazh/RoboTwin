
from .base_task import Base_task
from .utils import *
import sapien

class gpt_beaker_grasp(Base_task):
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
        tag = np.random.randint(0,2)
        if tag==0:
            self.beaker,_ = rand_create_glb(
                self.scene,
                xlim=[0.15,0.3],
                ylim=[0.13,0.15],
                zlim=[0.8],
                modelname="043_beaker",
                rotate_rand=False,
                qpos=[0.707,0.707,0,0],
                scale=(0.09,0.12,0.09),
            )

            beaker_pose = self.beaker.get_pose().p

            coaster_pose = rand_pose(
                xlim=[-0.05,0.1],
                ylim=[-0.2,0.05],
                zlim=[0.76],
                rotate_rand=False,
                qpos=[0.707,0.707,0,0],
            )

            while np.sum(pow(beaker_pose[:2] - coaster_pose.p[:2],2)) < 0.01:
                coaster_pose = rand_pose(
                    xlim=[-0.05,0.1],
                    ylim=[-0.2,0.05],
                    zlim=[0.76],
                    rotate_rand=False,
                    qpos=[0.707,0.707,0,0],
                )

            self.coaster,_ = create_obj(
                self.scene,
                pose = coaster_pose,
                modelname="019_coaster",
                convex=True
            )

            # self.coaster,_ = rand_create_glb(
            #     self.scene,
            #     xlim=[-0.05,0.1],
            #     ylim=[0.13,0.15],
            #     zlim=[0.76],
            #     modelname="019_coaster",
            #     rotate_rand=False,
            #     qpos=[0.707,0.707,0,0],
            #     scale=(0.55,0.55,0.55),
            # )
        else:
            self.beaker,_ = rand_create_glb(
                self.scene,
                xlim=[-0.3,-0.15],
                ylim=[0.13,0.15],
                zlim=[0.8],
                modelname="043_beaker",
                rotate_rand=False,
                qpos=[0.707,0.707,0,0],
                scale=(0.09,0.12,0.09),
            )

            beaker_pose = self.beaker.get_pose().p

            coaster_pose = rand_pose(
                xlim=[-0.05,0.1],
                ylim=[0.13,0.15],
                zlim=[0.76],
                rotate_rand=False,
                qpos=[0.707,0.707,0,0],
            )

            while np.sum(pow(beaker_pose[:2] - coaster_pose.p[:2],2)) < 0.01:
                coaster_pose = rand_pose(
                    xlim=[-0.05,0.1],
                    ylim=[0.13,0.15],
                    zlim=[0.76],
                    rotate_rand=False,
                    qpos=[0.707,0.707,0,0],
                )
            self.coaster,_ = create_obj(
                self.scene,
                pose = coaster_pose,
                modelname="019_coaster",
                convex=True,
                scale=(0.55,0.55,0.55),
            )

            # self.coaster,_ = rand_create_glb(
            #     self.scene,
            #     xlim=[-0.1, 0.05],
            #     ylim=[0.13,0.15],
            #     zlim=[0.76],
            #     modelname="019_coaster",
            #     rotate_rand=False,
            #     qpos=[0.707,0.707,0,0],
            #     scale=(0.09,0.12,0.09),
            # )

        self.beaker.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.01
        self.coaster.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.01

        render_freq = self.render_freq
        self.render_freq = 0
        for _ in range(4):
            self.together_open_gripper(save_freq=None)
        self.render_freq = render_freq

    
    # def play_once(self):
    #     # Retrieve the beaker and coaster objects
    #     beaker = self.actor_name_dic['beaker']
    #     coaster = self.actor_name_dic['coaster']
        
    #     # Retrieve the beaker and coaster data
    #     beaker_data = self.actor_data_dic['beaker_data']
    #     coaster_data = self.actor_data_dic['coaster_data']
        
    #     # Get the current pose of the beaker
    #     beaker_pose = self.get_actor_functional_pose(beaker, beaker_data)
        
    #     # Determine which arm to use based on the beaker's x-coordinate
    #     if beaker_pose[0] > 0:
    #         arm_tag = "right"
    #         move_function = self.right_move_to_pose_with_screw
    #         close_gripper_function = self.close_right_gripper
    #         open_gripper_function = self.open_right_gripper
    #     else:
    #         arm_tag = "left"
    #         move_function = self.left_move_to_pose_with_screw
    #         close_gripper_function = self.close_left_gripper
    #         open_gripper_function = self.open_left_gripper
        
    #     # Get the grasp pose for the beaker
    #     pre_grasp_pose = self.get_grasp_pose_to_grasp_object(endpose_tag=arm_tag, actor=beaker, actor_data=beaker_data, pre_dis=0.09)
    #     target_grasp_pose = self.get_grasp_pose_to_grasp_object(endpose_tag=arm_tag, actor=beaker, actor_data=beaker_data, pre_dis=0)
        
    #     # Move to the pre-grasp pose
    #     move_function(pre_grasp_pose)
        
    #     # Move to the target grasp pose
    #     move_function(target_grasp_pose)
        
    #     # Close the gripper to grasp the beaker
    #     close_gripper_function(pos=0.01)  # Adjust the position to ensure a gentle grasp
        
    #     # Lift the beaker slightly
    #     lift_pose = pre_grasp_pose.copy()
    #     lift_pose[2] += 0.1  # Lift the beaker by 10 cm
    #     move_function(lift_pose)
        
    #     # Get the target pose for placing the beaker on the coaster
    #     coaster_pose = self.get_actor_goal_pose(coaster, coaster_data, id=0)
    #     target_place_pose = self.get_grasp_pose_from_goal_point_and_direction(
    #         actor=beaker, actor_data=beaker_data, endpose_tag=arm_tag, actor_functional_point_id=1,
    #         target_point=coaster_pose, target_approach_direction=self.world_direction_dic['top_down'],
    #         actor_target_orientation=[0, 1, 0], pre_dis=0.09
    #     )
        
    #     # Move to the pre-place pose
    #     move_function(target_place_pose)
        
    #     # Move to the target place pose
    #     target_place_pose[2] -= 0.09  # Adjust the height to place the beaker on the coaster
    #     move_function(target_place_pose)
        
    #     # Open the gripper to place the beaker
    #     open_gripper_function()
        
    #     # Lift the arm slightly after placing the beaker
    #     lift_pose = target_place_pose.copy()
    #     lift_pose[2] += 0.1  # Lift the arm by 10 cm
    #     move_function(lift_pose)
    
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