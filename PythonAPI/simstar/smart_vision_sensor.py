import numpy 
import msgpackrpc

from .types import *
from .sensor import *

class SmartVisionSensor(Sensor):
    def __init__(self, client, ID):
        super().__init__(client, ID)
    
    def get_sensor_detections(self):
        return self.client.call("GetSmartVisionSensorDetections", self._ID)

    def get_lane_points(self):
        return self.client.call("GetVisionLanePoints", self._ID)