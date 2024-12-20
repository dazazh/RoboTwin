
from .base_task import Base_task
from .beaker_grasp import beaker_grasp
from .utils import *
import sapien

class gpt_beaker_grasp(beaker_grasp):
    def play_once(self):
        # Retrieve the beaker and coaster actors and their data
        beaker = self.actor_name_dic['beaker']
        coaster = self.actor_name_dic['coaster']
        beaker_data = self.actor_data_dic['beaker_data']
        coaster_data = self.actor_data_dic['coaster_data']

        # Get the current pose of the beaker
        beaker_pose = self.get_actor_functional_pose(beaker, beaker_data)

        # Determine which arm to use based on the beaker's x-coordinate
        if beaker_pose[0] > 0:
            # Use the right arm to grasp the beaker
            endpose_tag = "right"
            move_to_pose_with_screw = self.right_move_to_pose_with_screw
            close_gripper = self.close_right_gripper
            open_gripper = self.open_right_gripper
        else:
            # Use the left arm to grasp the beaker
            endpose_tag = "left"
            move_to_pose_with_screw = self.left_move_to_pose_with_screw
            close_gripper = self.close_left_gripper
            open_gripper = self.open_left_gripper

        # Get the pre-grasp and target grasp poses for the beaker
        pre_grasp_pose = self.get_grasp_pose_to_grasp_object(endpose_tag, beaker, beaker_data, pre_dis=0.05)
        target_grasp_pose = self.get_grasp_pose_to_grasp_object(endpose_tag, beaker, beaker_data, pre_dis=0)

        # Move to the pre-grasp pose
        move_to_pose_with_screw(pre_grasp_pose)

        # Move to the target grasp pose
        move_to_pose_with_screw(target_grasp_pose)

        # Close the gripper to grasp the beaker tightly
        close_gripper()

        # Lift the beaker slightly
        lift_pose = pre_grasp_pose.copy()
        lift_pose[2] += 0.1  # Lift the beaker by 0.1 meters
        move_to_pose_with_screw(lift_pose)

        # Get the target pose for placing the beaker on the coaster
        coaster_pose = self.get_actor_goal_pose(coaster, coaster_data, id=0)
        target_place_pose = self.get_grasp_pose_from_goal_point_and_direction(
            beaker, beaker_data, endpose_tag, actor_functional_point_id=1,
            target_point=coaster_pose, target_approach_direction=self.world_direction_dic['top_down'],
            actor_target_orientation=[0, 1, 0], pre_dis=0.05
        )

        # Move to the target place pose
        move_to_pose_with_screw(target_place_pose)

        # Open the gripper to place the beaker on the coaster
        open_gripper()

        # Move the arm back to the pre-grasp pose
        move_to_pose_with_screw(pre_grasp_pose)
