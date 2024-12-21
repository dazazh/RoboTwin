
from .base_task import Base_task
from .container_place import container_place
from .utils import *
import sapien

class gpt_container_place(container_place):
    def play_once(self):
        # Retrieve the actor and actor_data objects
        container = self.actor_name_dic['container']
        container_data = self.actor_data_dic['container_data']
        plate = self.actor_name_dic['plate']
        plate_data = self.actor_data_dic['plate_data']

        # Get the current pose of the container
        container_pose = self.get_actor_functional_pose(container, container_data)

        # Determine which arm to use based on the container's x coordinate
        if container_pose[0] > 0:
            # Use right arm to grasp the right side of the container
            endpose_tag = "right"
            arm_move_to_pose = self.right_move_to_pose_with_screw
            close_gripper = self.close_right_gripper
            open_gripper = self.open_right_gripper
        else:
            # Use left arm to grasp the left side of the container
            endpose_tag = "left"
            arm_move_to_pose = self.left_move_to_pose_with_screw
            close_gripper = self.close_left_gripper
            open_gripper = self.open_left_gripper

        # Get the grasp pose for the container
        pre_grasp_pose = self.get_grasp_pose_to_grasp_object(endpose_tag=endpose_tag, actor=container, actor_data=container_data, pre_dis=0.09)

        # Move the arm to the pre-grasp pose
        arm_move_to_pose(pre_grasp_pose)

        target_grasp_pose = self.get_grasp_pose_to_grasp_object(endpose_tag=endpose_tag, actor=container, actor_data=container_data, pre_dis=0)
        # Move the arm to the target grasp pose
        arm_move_to_pose(target_grasp_pose)

        # Close the gripper to grasp the container tightly
        close_gripper(pos=-0.01)

        # Lift the container up
        arm_move_to_pose(pre_grasp_pose)

        # Get the target pose for the plate
        plate_target_pose = self.get_actor_goal_pose(plate, plate_data, id=0)

        # Get the grasp pose to place the container on the plate
        place_pose = self.get_grasp_pose_from_goal_point_and_direction(container, container_data, endpose_tag=endpose_tag, actor_functional_point_id=0, target_point=plate_target_pose, target_approach_direction=self.world_direction_dic['top_down'], pre_dis=0.09)

        # Move the arm to the pre-place pose
        arm_move_to_pose(place_pose)

        target_place_pose = self.get_grasp_pose_from_goal_point_and_direction(container, container_data, endpose_tag=endpose_tag, actor_functional_point_id=0, target_point=plate_target_pose, target_approach_direction=self.world_direction_dic['top_down'], pre_dis=0)
        # Move the arm to the target place pose
        arm_move_to_pose(target_place_pose)

        # Open the gripper to place the container on the plate
        open_gripper()

        # Move the arm back to the pre-place pose
        arm_move_to_pose(place_pose)
