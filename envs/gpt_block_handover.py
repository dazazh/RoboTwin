
from .base_task import Base_task
from .block_handover import block_handover
from .utils import *
import sapien

class gpt_block_handover(block_handover):
    def play_once(self):
        # Retrieve actor objects and data
        grasp_block = self.actor_name_dic['grasp_block']
        target_block = self.actor_name_dic['target_block']
        handover_block_pose = self.actor_name_dic['handover_block_pose']
        
        grasp_block_data = self.actor_data_dic['grasp_block_data']
        target_block_data = self.actor_data_dic['target_block_data']
        handover_block_pose_data = self.actor_data_dic['handover_block_pose']
        
        # Get the target positions for the blocks
        grasp_block_pose = self.get_actor_goal_pose(grasp_block, grasp_block_data, id=0)
        target_block_pose = self.get_actor_goal_pose(target_block, target_block_data, id=0)
        handover_pose = self.get_actor_goal_pose(handover_block_pose, handover_block_pose_data, id=0)
        
        # Grasp the block with the left arm
        left_pre_grasp_pose = self.get_grasp_pose_to_grasp_object(endpose_tag="left", actor=grasp_block, actor_data=grasp_block_data, pre_dis=0.09)
        left_target_grasp_pose = self.get_grasp_pose_to_grasp_object(endpose_tag="left", actor=grasp_block, actor_data=grasp_block_data, pre_dis=0)
        
        self.left_move_to_pose_with_screw(left_pre_grasp_pose)
        self.left_move_to_pose_with_screw(left_target_grasp_pose)
        self.close_left_gripper()
        
        # Move the block to the handover point
        left_handover_target_pose = self.get_grasp_pose_from_goal_point_and_direction(grasp_block, grasp_block_data, endpose_tag="left", actor_functional_point_id=0, target_point=handover_pose, target_approach_direction=self.world_direction_dic['top_down'], pre_dis=0)
        
        self.left_move_to_pose_with_screw(left_handover_target_pose)
        
        # Handover the block to the right arm
        right_pre_grasp_pose = self.get_grasp_pose_to_grasp_object(endpose_tag="right", actor=grasp_block, actor_data=grasp_block_data, pre_dis=0.09)
        right_target_grasp_pose = self.get_grasp_pose_to_grasp_object(endpose_tag="right", actor=grasp_block, actor_data=grasp_block_data, pre_dis=0)
        
        self.right_move_to_pose_with_screw(right_pre_grasp_pose)
        self.right_move_to_pose_with_screw(right_target_grasp_pose)
        self.close_right_gripper()
        self.open_left_gripper()
        
        # Move the block to the target position
        right_target_pre_pose = self.get_grasp_pose_from_goal_point_and_direction(grasp_block, grasp_block_data, endpose_tag="right", actor_functional_point_id=0, target_point=target_block_pose, target_approach_direction=self.world_direction_dic['top_down'], pre_dis=0.09)
        right_target_target_pose = self.get_grasp_pose_from_goal_point_and_direction(grasp_block, grasp_block_data, endpose_tag="right", actor_functional_point_id=0, target_point=target_block_pose, target_approach_direction=self.world_direction_dic['top_down'], pre_dis=0)
        
        self.right_move_to_pose_with_screw(right_target_pre_pose)
        self.right_move_to_pose_with_screw(right_target_target_pose)
        self.open_right_gripper()
