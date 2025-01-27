
from .base_task import Base_task
from .beaker_pour_flask import beaker_pour_flask
from .utils import *
import sapien

class gpt_beaker_pour_flask(beaker_pour_flask):
    def play_once(self):
        # Retrieve the actor objects and their data
        beaker = self.actor_name_dic['beaker']
        conical_flask = self.actor_name_dic['conical_flask']
        beaker_data = self.actor_data_dic['beaker_data']
        conical_flask_data = self.actor_data_dic['conical_flask_data']

        # Get the initial pose of the beaker
        beaker_pose = self.get_actor_functional_pose(beaker, beaker_data)

        # Determine which arm to use based on the beaker's x coordinate
        if beaker_pose[0] > 0:
            arm_tag = "right"
            move_function = self.right_move_to_pose_with_screw
            close_gripper_function = self.close_right_gripper
            open_gripper_function = self.open_right_gripper
        else:
            arm_tag = "left"
            move_function = self.left_move_to_pose_with_screw
            close_gripper_function = self.close_left_gripper
            open_gripper_function = self.open_left_gripper

        # Get the grasp pose for the beaker
        pre_grasp_pose = self.get_grasp_pose_to_grasp_object(endpose_tag=arm_tag, actor=beaker, actor_data=beaker_data, pre_dis=0.09)
        target_grasp_pose = self.get_grasp_pose_to_grasp_object(endpose_tag=arm_tag, actor=beaker, actor_data=beaker_data, pre_dis=0)

        # Move to the pre-grasp pose
        move_function(pre_grasp_pose)

        # Move to the target grasp pose and close the gripper to grasp the beaker
        move_function(target_grasp_pose)
        close_gripper_function()

        # Lift the beaker high above the conical flask to avoid collisions
        lifted_pose = target_grasp_pose.copy()
        lifted_pose[2] += 0.2  # Lift the beaker 0.2 meters above its current position
        move_function(lifted_pose)

        # Get the functional pose of the conical flask
        conical_flask_functional_pose = self.get_actor_functional_pose(conical_flask, conical_flask_data)

        # Define the target point for pouring (above the conical flask)
        target_point = conical_flask_functional_pose[:3]
        target_point[2] += 0.1  # Position the beaker 0.1 meters above the conical flask

        # Define the approach direction for pouring (top_down)
        target_approach_direction = self.world_direction_dic['top_down']

        # Get the pose to move the beaker to the target point
        move_to_target_pose = self.get_grasp_pose_from_goal_point_and_direction(
            actor=beaker, actor_data=beaker_data, endpose_tag=arm_tag, actor_functional_point_id=0,
            target_point=target_point, target_approach_direction=target_approach_direction, pre_dis=0.09
        )

        # Move the beaker to the target point
        move_function(move_to_target_pose)

        # Tilt the beaker so that its spout is aligned perpendicularly with the mouth of the conical flask
        tilted_pose = move_to_target_pose.copy()
        tilted_pose[4] = -0.707  # Adjust the quaternion to tilt the beaker
        tilted_pose[6] = 0.707
        move_function(tilted_pose)

        # Open the gripper to pour the contents
        open_gripper_function()

        # Move the beaker back to the lifted position
        move_function(lifted_pose)

        # Move the beaker back to the initial position and release it
        move_function(target_grasp_pose)
        open_gripper_function()

        # Move the arm back to the pre-grasp pose
        move_function(pre_grasp_pose)
