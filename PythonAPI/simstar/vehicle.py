

import numpy 
import msgpackrpc

from .types import *
from .distance_sensor import *
from .radar_sensor import *
from .imu_sensor import *
from .smart_vision_sensor import *

class Vehicle():
    def __init__(self,client,ID):
        self._ID = int(ID)
        self.client = client

    def getID(self):
        return self._ID
    
    def set_controller_type(self, controller_type):
        self.client.call("SetControllerType", self._ID, controller_type)
    
    def control_vehicle(self,throttle,steer,brake):
        control = VehicleControl(throttle,steer,brake)
        self.client.call("ControlVehicle",control,self._ID)

    def add_distance_sensor(self, sensor_parameters):
        sensor_id = self.client.call("AddDistanceSensor", self._ID, sensor_parameters)
        return DistanceSensor(self.client, sensor_id)
        
    def add_radar_sensor(self, sensor_parameters):
        sensor_id = self.client.call("AddRadarSensor", self._ID, sensor_parameters)
        return RadarSensor(self.client, sensor_id)

    def add_imu_sensor(self):
        sensor_id = self.client.call("AddIMUSensor", self._ID)
        return ImuSensor(self.client, sensor_id)
        
    def add_smart_vision_sensor(self, sensor_parameters):
        sensor_id = self.client.call("AddSmartVisionSensor", self._ID, sensor_parameters)
        return SmartVisionSensor(self.client, sensor_id)

    # global frame
    def get_vehicle_state(self):
        return self.client.call("GetVehicleStateInfo",self._ID)

    # in vehicle frame
    def get_vehicle_state_self_frame(self):
        return self.client.call("GetVehicleStateInfoInSelfFrame",self._ID)

    def check_for_collision(self):
        return self.client.call("CheckForCollision",self._ID)

    def get_road_deviation_info(self):
        return self.client.call("GetRoadDeviationInfo",self._ID)
