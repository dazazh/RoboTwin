from .base_task import Base_task
from .utils import *
import sapien
from math import pi

class column_insert_WBCD(Base_task):
    def setup_demo(self, **kwags):
        super()._init(**kwags, table_static=True)
        self.create_table_and_wall()
        self.load_robot()
        self.setup_planner()
        self.load_camera()
        self.pre_move()
        self.load_actors()

        self.left_tube_mid_position = [-0.3,-0.32,0.935]
        self.right_tube_mid_position = [0.3,-0.32,0.935]
        self.step_lim = 500
        # print(f"./task_config/scene_info/{self.task_name}.json")
        # print(self.actor_name_dic)
        # print(self.actor_data_dic)

    def pre_move(self):
        render_freq = self.render_freq
        self.render_freq = 0
        self.together_open_gripper(save_freq=None)
        self.render_freq = render_freq

    def load_actors(self):
        self.down_column,self.down_column_data = rand_create_glb(
            self.scene,
            xlim=[0.1,0.25],
            ylim=[-0.1,0.1],
            zlim=[0.76],
            modelname="049_down_column",
            convex=False,
            rotate_rand=False,
            rotate_lim=[0,0,1.57],
            is_static=False,
            qpos=[1,0,0,0],
        )

        self.up_column, self.up_column_data = rand_create_glb(
            self.scene,
            xlim=[-0.25,-0.1],
            ylim=[-0.1,0.1],
            zlim=[0.745],
            modelname="050_up_column",
            rotate_rand=False,
            rotate_lim=[0,0,1.57],
            convex=True,
            qpos=[1,0,0,0],
            is_static=False,
        )
        
        self.up_column.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.01
        self.down_column.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.01
    def play_once(self):
        
        # Get the grasp pose for the beaker
        up_pre_grasp_pose = self.get_grasp_pose_w_labeled_direction(actor=self.up_column, actor_data=self.up_column_data, grasp_matrix=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]), pre_dis=0.1, id=0)
        down_pre_grasp_pose = self.get_grasp_pose_w_labeled_direction(actor=self.down_column, actor_data=self.down_column_data, grasp_matrix=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]), pre_dis=0.1, id=0)
        
        up_target_grasp_pose = self.get_grasp_pose_w_labeled_direction(actor=self.up_column, actor_data=self.up_column_data, grasp_matrix=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]), pre_dis=0.03, id=0)
        down_target_grasp_pose = self.get_grasp_pose_w_labeled_direction(actor=self.down_column, actor_data=self.down_column_data, grasp_matrix=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]), pre_dis=0.03, id=0)

        up_lift_grasp_pose = up_pre_grasp_pose.copy()
        up_lift_grasp_pose[2] += 0.05
        up_lift_grasp_pose[3:]=[-0.924, 0,   0,    -0.383]

        down_lift_grasp_pose = down_pre_grasp_pose.copy()
        down_lift_grasp_pose[2] += 0.05
        down_lift_grasp_pose[3:]=[-0.383, 0,   0,    -0.924]

        left_mid_pose = [-0.19,-0.12,0.92,1,0,0,0]
        right_mid_pose = [0.19,-0.12,0.92,-0.01,0.01,0.03,-1]
        left_target_pose = [-0.09,-0.1,1.08,1,0,0,0]
        # right_target_pose = [0.09,-0.1,0.88,-0.01,0.01,0.03,-1]

        # Move to the pre-grasp pose
        self.together_move_to_pose_with_screw(left_target_pose=up_pre_grasp_pose,right_target_pose=down_pre_grasp_pose)
        
        # Move to the target grasp pose
        self.together_move_to_pose_with_screw(left_target_pose=up_target_grasp_pose,right_target_pose=down_target_grasp_pose)
        
        # Close the gripper to grasp the beaker
        # Adjust the position to ensure a gentle grasp
        self.together_close_gripper(left_pos=-0.05,right_pos=-0.05)

        up_pre_grasp_pose_ = up_pre_grasp_pose.copy()
        up_pre_grasp_pose_[2] += 0.1
        self.together_move_to_pose_with_screw(left_target_pose=up_pre_grasp_pose,right_target_pose=down_pre_grasp_pose)
        self.together_move_to_pose_with_screw(left_target_pose=left_mid_pose,right_target_pose=right_mid_pose)
        
        # self.together_move_to_pose_with_screw(left_target_pose=left_target_pose,right_target_pose=right_target_pose)
        self.left_move_to_pose_with_screw(left_target_pose)

        right_target_pose_p = self.get_actor_goal_pose(self.up_column, self.up_column_data)[:3]
        print(self.up_column.get_pose().q)
        # right_target_pose_q = self.get_pose_q(self.up_column.get_pose().q)
        # print(right_target_pose_q)
        right_target_pose_q = [-0.01,0.01,0.03,-1]

        right_target_pose = self.get_target_pose_from_goal_point_and_direction(actor=self.down_column, actor_data=self.down_column_data, endpose=self.right_endpose,target_pose=right_target_pose_p, target_grasp_qpose=right_target_pose_q)
        # right_target_pose = list(right_target_pose) + [-0.01,0.01,0.03,-1]
        # print(left_target_pose) [0.463,0.514,-0.543,-0.486]
        right_target_pose[0]+=0.0135
        right_target_pose[2]-=0.01
        self.right_move_to_pose_with_screw(right_target_pose)

        left_insert_pose = left_target_pose.copy()
        left_insert_pose[2] -= 0.01
        self.left_move_to_pose_with_screw(left_insert_pose)

    # Check success
    def check_success(self):
        return True
        # target_point = self.get_actor_contact_position(self.test_tube_rack, self.test_tube_rack_data, id = 9)
        # place_point = self.get_actor_goal_pose(self.test_tube, self.test_tube_data)[:3]
        # eps = 0.05
        # return abs(place_point[0] - target_point[0])<eps  and  abs(place_point[1] - target_point[1])<eps and abs(place_point[2] - target_point[2])<eps
        