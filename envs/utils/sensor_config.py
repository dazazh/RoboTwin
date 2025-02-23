from sapien.sensor import StereoDepthSensorConfig
import numpy as np

class SensorConfig(StereoDepthSensorConfig):
    def __init__(self):
        super().__init__()
        # ================= RGB 参数 =================
        self.rgb_resolution = (320, 240)  # 新RGB分辨率
        # RGB内参缩放比例（保持原视场角）
        rgb_scale_x = 320 / 1920  # 原宽度1920 → 新宽度320
        rgb_scale_y = 240 / 1080  # 原高度1080 → 新高度240
        
        self.rgb_intrinsic = np.array([
            [1380.0 * rgb_scale_x, 0.0,           960.0 * rgb_scale_x],  # fx=230.0, cx=160.0
            [0.0,           1380.0 * rgb_scale_y, 540.0 * rgb_scale_y],  # fy=306.67, cy=120.0
            [0.0,           0.0,           1.0]
        ])

        # ================= 红外参数 =================
        # self.ir_resolution = (320, 240)  # 同步红外分辨率
        # ir_scale_x = 320 / 1280  # 原宽度1280 → 新宽度320
        # ir_scale_y = 240 / 720   # 原高度720 → 新高度240
        
        # self.ir_intrinsic = np.array([
        #     [920.0 * ir_scale_x, 0.0,         640.0 * ir_scale_x],  # fx=230.0, cx=160.0
        #     [0.0,         920.0 * ir_scale_y, 360.0 * ir_scale_y],  # fy≈306.67, cy=120.0
        #     [0.0,         0.0,         1.0]
        # ])

        # ================= 其他关键参数 =================
        self.depth_dilation = True  # 建议保持开启（分辨率降低后更需要补洞）
        self.median_filter_size = 5  # 可适当增大滤波尺寸（低分辨率需要更强降噪）
        self.block_width = 5         # 可缩小匹配窗口（原7对于320x240可能过大）
        self.block_height = 5