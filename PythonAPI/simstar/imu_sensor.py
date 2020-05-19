import numpy 
import msgpackrpc

from .types import *
from .sensor import *

class ImuSensor(Sensor):
    def __init__(self, client, ID):
        super().__init__(client, ID)
    
    def get_sensor_detections(self):
        return self.client.call("GetIMUMeasurements", self._ID)
