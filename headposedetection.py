
class HeadPoseDetection:
    def __init__(self):
        self.x_min = 0
        self.y_min = 0
        self.x_max = 0
        self.y_max = 0
        self.bbox_height = 0
        self.bbox_width = 0
        self.confidence = 0
        self.cropped_clr_img = None
        self.cropped_img = None
        self.yaw = 0
        self.pitch = 0
        self.roll = 0