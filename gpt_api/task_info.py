# All variable names for task information must be in uppercase.

# Template of Task Information:
"""
TASK_NAME = {
    'task_name': "{task_name}",
    "task_description": "...",
    "current_code": '''
                class gpt_{task_name}({task_name}):
                    def play_once(self):
                        pass
                '''
}
"""

################## Known Tasks ##################

APPLE_CABINET_STORAGE = {
    "task_name": "apple_cabinet_storage",
    "task_description": "Grab the handle of the cabinet with your left arm and open the cabinet while grabbing the apple on the table with your right arm. \
                        Then use your right arm to lift up the apple and put the apple in the cabinet and use your left arm to close the cabinet.\
                        Note: You need to call the get_avoid_collision_pose function on the right arm before closing the cabinet to avoid the cabinet colliding with the right arm.\
                              The direction of pulling the cabinets apart is the negative direction of the y-axis, and pull-out drawer 18cm.\
                              You may need to close the clamp jaws very tightly to open the cabinet.",
    "current_code": '''
                class gpt_apple_cabinet_storage(apple_cabinet_storage):
                    def play_once(self):
                        pass
                '''
}

BLOCK_HAMMER_BEAT = {
    "task_name": "block_hammer_beat",
    "task_description": "Pick up the hammet and use it to beat the block on the table once. The hammer is placed at a fixed position on the table, \
                        but the block is generated randomly on the table, if block's x coordinate (dim 0) is greater than 0, use right arm the grasp the hammer, \
                        else use the left arm. Then use the hammer to beat the block, and the hammer does not need to be lifted high again.\
                        Note: You don't need to put the hammer down.",
    "current_code": '''
                class gpt_block_hammer_beat(block_hammer_beat):
                    def play_once(self):
                        pass
                '''
}


DUAL_BOTTLES_PICK_EASY = {
    "task_name": "dual_bottles_pick_easy",
    "task_description": "Use both arms to simultaneously pick up the red and green bottles and move them to the front target locations, \
                        with the red bottle on the left and the green bottle on the right.\
                        Note: You don't need to open gripper and don't put down the bottles at the end.",
    "current_code": '''
                class gpt_dual_bottles_pick_easy(dual_bottles_pick_easy):
                    def play_once(self):
                        pass
                '''
}   


DUAL_BOTTLES_PICK_HARD = {
    "task_name": "dual_bottles_pick_hard",
    "task_description": "Use both arms to simultaneously pick up the red and green bottles and move them to the front target locations, \
                        with the red bottle on the left and the green bottle on the right. No need to put the bottles down. In which the bottles may be lying down.\
                        Note: You don't need to open gripper and don't put down the bottles at the end.\
                              The clamping jaw needs to be lifted upward after grabbing the bottle.",
    "current_code": '''
                class gpt_dual_bottles_pick_hard(dual_bottles_pick_hard):
                    def play_once(self):
                        pass
                '''
}

DIVERSE_BOTTLES_PICK = {
    "task_name": "diverse_bottles_pick",
    "task_description": "Use both arms to simultaneously pick up the diverse bottles and move them to the front target locations, \
                        with the bottle1 on the left and the bottle2 on the right. No need to put the bottles down. In which the bottles may be lying down.\
                        Note: You don't need to open gripper and don't put down the bottles at the end.",
    "current_code": '''
                class gpt_diverse_bottles_pick(diverse_bottles_pick):
                    def play_once(self):
                        pass
                '''
}

BLOCK_HANDOVER = {
    "task_name": "block_handover",
    "task_description": "There are two blocks on the desk. Use the left arm to grab the block and move it to the handover point, then use right arm to grab the block and open the gripper of left arm simutaniously.\
                        Use right arm move block on the target block.\
                        Note: You should first pass the block to the right gripper and close right gripper, then open the left gripper.",
    "current_code": '''
                class gpt_block_handover(block_handover):
                    def play_once(self):
                        pass
                '''
}

BLOCKS_STACK_EASY = {
    "task_name": "blocks_stack_easy",
    "task_description": "Use the gripper to pick up block1 and move block 1 to the target position. Then pick up block 2 and place it on the block 1.\
                        If block1's x coordinate (dim 0) is greater than 0, use right arm to stack the block1, else use the left arm, and same for the block2.\
                        Note: You need to call the get_avoid_collision_pose function to avoid collisions when the left and right arms move alternately. \
                              For example, if the previous action uses the left arm and the next action uses the right arm, you need to move the left arm after release gripper to avoid collisions, vice versa.\
                              The pre-dis of stacked blocks may be smaller.",
    "current_code": '''
                class gpt_blocks_stack_easy(blocks_stack_easy):
                    def play_once(self):
                        pass
                '''
}

BLOCKS_STACK_HARD = {
    "task_name": "blocks_stack_hard",
    "task_description": "Use the gripper to pick up block1 and move block 1 to the target position. Then pick up block 2 and place it on the block 1, and finally pick up\
                        block3 and place it on the block2.\
                        If block1's x coordinate (dim 0) is greater than 0, use right arm to stack the block1, else use the left arm. And same for the block2 and block3.\
                        Note: You need to call the get_avoid_collision_pose function to avoid collisions when the left and right arms move alternately. \
                              For example, if the previous action uses the left arm and the next action uses the right arm, you need to move the left arm after release gripper to avoid collisions, vice versa.\
                              The pre-dis of stacked blocks may be smaller.",
    "current_code": '''
                class gpt_blocks_stack_hard(blocks_stack_hard):
                    def play_once(self):
                        pass
                '''
}

CONTAINER_PLACE = {
    "task_name": "container_place",
    "task_description": "Use both arms to pick up the container and place it in the plate. If the container's x coordinate (dim 0) is greater than 0, \
                        use right arm to grasp the right side of the container, then pick up the container and place it in the plate. \
                        Else use the left arm grasp the left side of the container, then pick up the container and place it in the plate.\
                        Note: You may need to close the jaws tightly to pick up the container.",
    "current_code": '''
                class gpt_container_place(container_place):
                    def play_once(self):
                        pass
                '''
}

EMPTY_CUP_PLACE = {
    "task_name": "empty_cup_place",
    "task_description": "Use both arms to pick up the empty cup and place it on the coaster. If the cup's x coordinate (dim 0) is greater than 0, \
                        use right arm to grasp the cup, then pick up the cup and place it on the coaster,\
                        else use the left arm grasp the the cup, then pick up the cup and place it on the coaster.\
                        Note: You may need to close the jaws tightly to pick up the cup.\
                              Pre-dis for grabbing and placing cups may be smaller.\
                              The distance of lifting the cup may be smaller.",
    "current_code": '''
                class gpt_empty_cup_place(empty_cup_place):
                    def play_once(self):
                        pass
                '''
}

MUG_HANGING = {
    "task_name": "mug_hanging",
    "task_description": "There is a mug on the left side of the table and a racker on the right side. Use your left arm to move the mug to a middle position,\
                        and and put the mug down. Then use your right arm to grab the mug and hang the mug on the racker's bracket.\
                        Note: You may need to avoid left arm collisions after open the left gripper.\
                              You can get middle_pos of left arm, and you can just move your left arm to this position.\
                              And you may need to close the jaws tightly to pick up the mug.",
    "current_code": '''
                class gpt_mug_hanging(mug_hanging):
                    def play_once(self):
                        pass
                '''
}

SHOE_PLACE = {
    "task_name": "shoe_place",
    "task_description": "Pick up the shoe and place it on the target block. And the head of the shoe should be towards the left side.\
                        The shoe is randomly placed on the table, if the shoe's x coordinate (dim 0) is greater than 0, use right arm to grasp the shoe, \
                        else use the left arm grasp the shoe.",
    "current_code": '''
                class gpt_shoe_place(shoe_place):
                    def play_once(self):
                        pass
                '''
}

SHOES_PLACE = {
    "task_name": "shoes_place",
    "task_description": "Left shoe and right shoe are randomly generated on the desktop, one on the left and one on the right.\
                        Use left and right arms to pick up two shoes simultaneously. And put down them on the target block respectively.\
                        The head of the shoe should be towards the left side.\
                        Left shoe should be placed on the point0 of target block, and right shoe should be placed on the point1 of target block.\
                        Note: You may need to put the shoes in order to avoid left and right arm collisions.\
                              Avoiding collisions needs to be done before place shoes.\
                              Pre-dis for grabbing and placing shoes may be smaller.",
    "current_code": '''
                class gpt_shoes_place(shoes_place):
                    def play_once(self):
                        pass
                '''
}

PICK_APPLE_MESSY = {
    "task_name": "pick_apple_messy",
    "task_description": "Pick up the apple on the desk and lift it up. If the apple's x coordinate (dim 0) is greater than 0, use right arm to grasp the apple, \
                        else use the left arm.",
    "current_code": '''
                class gpt_pick_apple_messy(pick_apple_messy):
                    def play_once(self):
                        pass
                ''' 
}

# BEAKER_GRASP = {
#     "task_name": "beaker_grasp",
#     "task_description": "Pick up the beaker and place it on the table. If the beaker's x coordinate (dim 0) is greater than 0, use right arm to grasp the beaker, \
#                         else use the left arm.",
#     "current_code": '''
#                 class gpt_beaker_grasp(beaker_grasp):
#                     def play_once(self):
#                         pass
#                 '''
# }

BEAKER_GRASP = {
    "task_name": "beaker_grasp",
    "task_description": "Use both arms to pick up the beaker from the lateral side and place it on the coaster. If the beaker's x coordinate (dim 0) is greater than 0, \
                        use right arm to grasp the beaker, then pick up the beaker and place it on the coaster,\
                        else use the left arm grasp the the beaker, then pick up the beaker from the lateral side and place it on the coaster.\
                        Note: Do not close the gripper too tight to pick up the beaker,the degree of closure shold be\
                                slightly smaller than the diameter of the beaker for better suitablity.",
    "current_code": '''
                class gpt_beaker_grasp(beaker_grasp):
                    def play_once(self):
                        pass
                '''
}