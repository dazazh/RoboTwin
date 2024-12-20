
from .base_task import Base_task
from .block_hammer_beat import block_hammer_beat
from .utils import *
import sapien

class gpt_block_hammer_beat(block_hammer_beat):
    def play_once(self):
        # Retrieve the actor objects
        hammer = self.actor_name_dic['hammer']
        block = self.actor_name_dic['block']

        # Retrieve the actor data objects
        hammer_data = self.actor_data_dic['hammer_data']
        block_data = self.actor_data_dic['block_data']

        # Get the block's position
        block_pose = self.get_actor_goal_pose(block, block_data, id=0)

        # Determine which arm to use based on the block's x coordinate
        if block_pose[0] > 0:
            arm_tag = "right"
            move_function = self.right_move_to_pose_with_screw
            close_gripper_function = self.close_right_gripper
            open_gripper_function = self.open_right_gripper
        else:
            arm_tag = "left"
            move_function = self.left_move_to_pose_with_screw
            close_gripper_function = self.close_left_gripper
            open_gripper_function = self.open_left_gripper

        # Get the hammer's pose
        hammer_pose = self.get_actor_goal_pose(hammer, hammer_data, id=0)

        # Get the grasp pose for the hammer
        pre_grasp_pose = self.get_grasp_pose_to_grasp_object(endpose_tag=arm_tag, actor=hammer, actor_data=hammer_data, pre_dis=0.09)
        target_grasp_pose = self.get_grasp_pose_to_grasp_object(endpose_tag=arm_tag, actor=hammer, actor_data=hammer_data, pre_dis=0)

        # Move to the pre-grasp pose
        move_function(pre_grasp_pose)

        # Move to the target grasp pose
        move_function(target_grasp_pose)

        # Close the gripper to grasp the hammer
        close_gripper_function()

        # Move back to the pre-grasp pose to lift the hammer
        move_function(pre_grasp_pose)

        # Get the target pose to beat the block
        target_beat_pose = self.get_grasp_pose_from_goal_point_and_direction(
            actor=hammer, actor_data=hammer_data, endpose_tag=arm_tag, actor_functional_point_id=0,
            target_point=block_pose, target_approach_direction=self.world_direction_dic['top_down'], pre_dis=0.05
        )

        # Move the hammer to the block's position
        move_function(target_beat_pose)

        # Move the hammer down to beat the block
        move_function(target_grasp_pose)

        # Move the hammer back up slightly after beating the block
        move_function(target_beat_pose)

        # Optionally, you can leave the hammer in place without putting it down
        # If you want to put the hammer down, you would open the gripper here
        # open_gripper_function()
