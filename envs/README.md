
Basic code structure for the task:
```python
from .base_task import Base_task
from .utils import *
import sapien

class ${task_name}(Base_task):
    def setup_demo(self, **kwargs):
        super()._init(**kwargs, table_static=True)
        self.create_table_and_wall()
        self.load_robot()
        self.setup_planner()
        self.load_camera(kwargs.get('camera_w', 640), kwargs.get('camera_h', 480))
        self.pre_move()
        self.load_actors(f"./task_config/scene_info/{self.task_name[4:]}.json")

    def pre_move(self):
        render_freq = self.render_freq
        self.render_freq = 0
        self.together_open_gripper(save_freq=None)
        self.render_freq = render_freq
        
    def play_once(self):
        pass
    
    # Check success
    def check_success(self):
        ...
```
In the code above, `{task_name}` should match the name of the basic code file, and the `check_success()` function is used to determine if the task is successful. No changes are needed for the rest of the code.
