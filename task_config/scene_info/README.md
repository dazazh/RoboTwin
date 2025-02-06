```json
{
    "actor_info_list": [
        {
            "actor_name": "...",
            "actor_file": "...",    // The name of the digital asset corresponding to the model in the models folder
            "actor_type": "glb",    // Type of digital asset, currently available types are glb, obj, urdf, box
            "x_limit": [[-0.25, -0.05], [0.05, 0.25]],  // X-coordinate limits for object generation
            "y_limit": [[-0.2, 0.05]],                // Y-coordinate limits for object generation
            "z_limit": [[0.78]],                     // Z-coordinate limits for object generation
            "qpose": [[0.7071,0,0,-0.7071]],                 // Initial orientation of the object as a quaternion
            "is_static_or_fix_root_link": false,     // Whether the object is fixed
            "ylim_prop": true,               // Protection for the generated y-coordinate range to prevent object generation
            "rotate_tag": false,            // Whether the object rotates randomly
            "rotate_limit": [],              // The range of random rotation of the object on the xyz axes
            "convex": true,                  // Whether the object's collision volume is a convex hull
            "z_val_protect": true,           // Set to true when generating diverse objects
            "actor_data_from": "file",       // Source of model data, either 'file' or a custom function name
            "rand_model_id": false,         // Whether to generate diverse models
            "mass": 0.01                    // Model mass
        },
        //...
    ],
    "rand_create_actor": true,           // For generating non-critical models for a cluttered desktop
    "rand_actor_num": 4,                 // Number of non-critical models generated
    "rand_choice_actor_list": [          // List of models for a cluttered desktop
        {
            "actor_name": "...",
            "actor_file": "...",
            "actor_type": "glb",
            "x_limit": [[-0.35, 0.35]],
            "y_limit": [[-0.15, 0.3]],
            "z_limit": [[0.8]],
            "qpose": [[-0.5, 0.5, 0.5, 0.5]],
            "is_static_or_fix_root_link": false,
            "ylim_prop": false,
            "rotate_tag": true,
            "rotate_limit": [[1.57, 0, 0]],
            "convex": false,
            "z_val_protect": true,
            "actor_data_from": "file",
            "rand_model_id": false,
            "mass": 0.01
        },
        //...
    ],
    "actor_dis_lim_mat": [[0, 0.2, 0.2, 0.2, 0.2],   // Matrix to limit the minimum distance between objects
                         [0.2, 0, 0.2, 0.2, 0.2],
                         [0.2, 0.2, 0, 0.2, 0.2],
                         [0.2, 0.2, 0.2, 0, 0.2],
                         [0.2, 0.2, 0.2, 0.2, 0]],
    "actor_position_link": false,   // Whether different object poses are linked
    "actor_position_num": 0,         // Number of random poses for different objects
    "other_actor_data": []          // List of other potentially used pose information
}
```

**Notes:**

1. In most cases, you only need to pay attention to the fields:

   `actor_name`, `actor_file`, `actor_type`, `x_limit`, `y_limit`, `z_limit`, `qpose`, `is_static_or_fix_root_link`, and `actor_dis_lim_mat`.
2. When `actor_type` is `box`, `actor_data_from` must be a custom function, refer to [blocks_stack_easy](../../envs/blocks_stack_easy.py) and [block_handover](../../envs/block_handover.py) files.
3. If `rand_model_id` is True, you need to define `self.id_list` in the corresponding task code.
4. `rand_create_actor` is generally False, and when it is False, `rand_choice_actor_list` is not needed;
5. The `actor_dis_lim_mat` matrix can be empty, in which case there is no restriction on the distance between objects;
6. `other_actor_data` is used to store other potentially used pose information, and you need to define the corresponding variable names in the file, refer to [dual_bottles_pick_easy.json](./dual_bottles_pick_easy.json) file.
