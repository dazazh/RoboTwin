
from .base_task import Base_task
from .pick_apple_messy import pick_apple_messy
from .utils import *
import sapien

class gpt_pick_apple_messy(pick_apple_messy):
    def play_once(self):
        # Retrieve the actor and actor_data
        apple = self.actor_name_dic['apple']
        apple_data = self.actor_data_dic['apple_data']

        # Get the apple's pose
        apple_pose = self.get_actor_functional_pose(apple, apple_data)

        # Determine which arm to use based on the apple's x coordinate
        if apple_pose[0] > 0:
            arm_tag = "right"
            move_function = self.right_move_to_pose_with_screw
            close_gripper_function = self.close_right_gripper
        else:
            arm_tag = "left"
            move_function = self.left_move_to_pose_with_screw
            close_gripper_function = self.close_left_gripper

        # Get the grasp pose for the apple
        pre_grasp_pose = self.get_grasp_pose_to_grasp_object(endpose_tag=arm_tag, actor=apple, actor_data=apple_data, pre_dis=0.09)
        target_grasp_pose = self.get_grasp_pose_to_grasp_object(endpose_tag=arm_tag, actor=apple, actor_data=apple_data, pre_dis=0)

        # Move the arm to the pre-grasp pose
        move_function(pre_grasp_pose)

        # Move the arm to the grasp pose
        move_function(target_grasp_pose)

        # Close the gripper to grasp the apple
        close_gripper_function()

        # Lift the apple by moving back to the pre-grasp pose
        move_function(pre_grasp_pose)
