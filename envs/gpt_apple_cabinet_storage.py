
from .base_task import Base_task
from .apple_cabinet_storage import apple_cabinet_storage
from .utils import *
import sapien

class gpt_apple_cabinet_storage(apple_cabinet_storage):
    def play_once(self):
        # Retrieve the actor objects
        cabinet = self.actor_name_dic['cabinet']
        apple = self.actor_name_dic['apple']

        # Retrieve the actor_data objects
        cabinet_data = self.actor_data_dic['cabinet_data']
        apple_data = self.actor_data_dic['apple_data']

        # Step 1: Grab the handle of the cabinet with the left arm
        handle_grasp_pose = self.get_grasp_pose_to_grasp_object(endpose_tag="left", actor=cabinet, actor_data=cabinet_data, pre_dis=0.09)
        handle_target_grasp_pose = self.get_grasp_pose_to_grasp_object(endpose_tag="left", actor=cabinet, actor_data=cabinet_data, pre_dis=0)

        self.left_move_to_pose_with_screw(handle_grasp_pose)
        self.left_move_to_pose_with_screw(handle_target_grasp_pose)
        self.close_left_gripper(pos=-0.02)  # Tighten the gripper to ensure it grabs the handle tightly

        # Step 2: Open the cabinet by pulling the handle in the negative y-axis direction (16cm)
        pull_distance = 0.18  # 16cm in meters
        handle_pull_pose = handle_target_grasp_pose.copy()
        handle_pull_pose[1] -= pull_distance  # Move in the negative y-axis direction

        self.left_move_to_pose_with_screw(handle_pull_pose)

        # Step 3: Grab the apple on the table with the right arm
        apple_grasp_pose = self.get_grasp_pose_to_grasp_object(endpose_tag="right", actor=apple, actor_data=apple_data, pre_dis=0.09)
        apple_target_grasp_pose = self.get_grasp_pose_to_grasp_object(endpose_tag="right", actor=apple, actor_data=apple_data, pre_dis=0)

        self.right_move_to_pose_with_screw(apple_grasp_pose)
        self.right_move_to_pose_with_screw(apple_target_grasp_pose)
        self.close_right_gripper()

        # Step 4: Lift the apple and place it inside the cabinet
        apple_lift_pose = apple_target_grasp_pose.copy()
        apple_lift_pose[2] += 0.2  # Lift the apple 20cm above the table
        self.right_move_to_pose_with_screw(apple_lift_pose)

        # Move the apple to the cabinet's functional point (inside the cabinet)
        cabinet_functional_point = self.get_actor_goal_pose(cabinet, cabinet_data, id=0)
        apple_inside_cabinet_pose = self.get_grasp_pose_from_goal_point_and_direction(apple, apple_data, endpose_tag="right", actor_functional_point_id=0, target_point=cabinet_functional_point, target_approach_direction=self.world_direction_dic['top_down'], pre_dis=0.09)
        # apple_inside_cabinet_target_pose = self.get_grasp_pose_from_goal_point_and_direction(apple, apple_data, endpose_tag="right", actor_functional_point_id=0, target_point=cabinet_functional_point, target_approach_direction=self.world_direction_dic['top_down'], pre_dis=0)

        self.right_move_to_pose_with_screw(apple_inside_cabinet_pose)
        # self.right_move_to_pose_with_screw(apple_inside_cabinet_target_pose)
        self.open_right_gripper()  # Release the apple inside the cabinet

        # Step 5: Close the cabinet with the left arm
        # First, move the right arm to a safe position to avoid collision
        right_avoid_collision_pose = self.get_avoid_collision_pose(avoid_collision_arm_tag='right')
        self.right_move_to_pose_with_screw(right_avoid_collision_pose)

        # Now close the cabinet
        handle_close_pose = handle_target_grasp_pose.copy()
        self.left_move_to_pose_with_screw(handle_close_pose)
