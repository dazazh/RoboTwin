
```yml
task_name: ...          # Task name
render_freq: 5          # Rendering window frequency, set to 0 to disable rendering window
use_seed: false         # Whether to use an existing seed list for data collection
collect_data: true      # Whether data collection is required
save_path: ./data       # Data save path
dual_arm: true          # Whether it is a dual-arm task
st_episode: 0           # Starting episode number for data collection
camera_w: 320           # Camera width and height
camera_h: 240
pcd_crop: true          # Whether to crop the point cloud to the table
pcd_down_sample_num: 1024   # Point cloud downsampling
episode_num: 100        # Number of episodes to collect
save_freq: 15           # Data save frequency
head_camera_fovy: 45    # Field of view angle of the top camera
save_type:              # Data save format
  raw_data: false
  pkl: true
data_type:              # Data content to save
  rgb: true
  observer: false
  depth: true
  pointcloud: true
  combine: false
  endpose: true
  qpos: true
  mesh_segmentation: false
  actor_segmentation: false
```

**Notes:**

1. `render_freq` is generally set to a multiple of 5.
2. If `use_seed` is True, there should be corresponding seed files in the `seeds/` folder.
3. `episode_num` is the last episode number and is not affected by `st_episode`.
