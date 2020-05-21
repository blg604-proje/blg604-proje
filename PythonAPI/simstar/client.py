try:
    import msgpackrpc
except ImportError:
    raise ImportError('pip install msgpack-rpc-python')

try:
    import numpy as np
except ImportError:
    raise ImportError('pip install numpy')

from .types import *
from .vehicle import * 
from .road import *
from .road_network_generator import *

import sys
import os
import time 
import math
import logging
import sys
import argparse
import enum

class Client():
    def __init__(self,host="127.0.0.1",port=8080,carlaport=2000):
        self.host = host
        self.port = port 
        self.client = msgpackrpc.Client(msgpackrpc.Address(host, port), timeout = 10, \
                pack_encoding = 'utf-8', unpack_encoding = 'utf-8')

    def ping(self):
        pong = self.client.call("ping")
        return pong

    def create_road_generator(self, spawn_location = WayPoint(0.0, 0.0, 0.0), 
            spawn_rotation = WayPointOrientation(0.0, 0.0, 0.0), number_of_lanes=3):
        road_id =  self.client.call("SpawnRoad", spawn_location,
            spawn_rotation, number_of_lanes)
        time.sleep(0.5)
        return RoadGenerator(self.client,road_id)

    def reset_level(self):
        self.client.call("OpenHighway")
        self.client.close()
        del self.client
        time.sleep(3) 
        self.client = msgpackrpc.Client(msgpackrpc.Address(self.host, \
                    self.port), timeout = 5, \
                pack_encoding = 'utf-8', unpack_encoding = 'utf-8')
        
    def spawn_vehicle(self,actor=0,distance=20,lane_id=1,initial_speed=30,set_speed=50,trajectory_data=[],vehicle_type=0):
        if actor == 0:
            vehicle_id = self.client.call("SpawnVehicle", False, distance, lane_id, initial_speed, set_speed, trajectory_data)
        else:
            vehicle_id = self.client.call("SpawnVehicleRelativeTo", actor.getID(), distance, lane_id, initial_speed, set_speed, trajectory_data, vehicle_type)
        return Vehicle(self.client,vehicle_id)
        
    def display_vehicle(self, vehicle):
        vehicle_id = vehicle.getID()
        self.client.call("DisplayVehicle", vehicle_id)

    def enable_lane_change(self,enable=True):
        self.client.call("EnableLaneChange", enable)

    def get_vehicle_ground_truths(self,in_ego_frame=False):
        vehicle_positions = []
        vehicle_positions = self.client.call("GetVehicleGroundTruths")
        return vehicle_positions

    def getSimulatorVersion(self):
        version = self.client.call('GetVersion')
        return version

    def changeControlHelperOption(self,option=0):
    	self.client.call("ChangeOptionACCorLCC",option)

    def autoPilotAgents(self, agents):
        agent_ids = []
        for agent in agents:
            agent_ids.append(agent.getID())
        self.client.call("AutoPilotAgents",agent_ids)
    
    def removeActors(self, actors):
        actor_ids = []
        for actor in actors:
            actor_ids.append(actor.getID())
        self.client.call("DeleteActors",actor_ids)

    def set_custom_model_flags(self,agent_flag,ego_flag=False):
        self.client.call("SetCustomModelUseFlags",ego_flag,agent_flag)

    def spawn_infinite_highway(self,is_curved, min_radius,
         is_traffic_enabled,interval_time,vehicle):
        vehicle_ID = vehicle.getID()
        self.client.call("InfiniteHighway", is_curved, min_radius, 
            is_traffic_enabled,interval_time,vehicle_ID)

    def check_for_all_vehicle_accidents(self):
        did_accident_occur = self.client.call("IsVehicleCollided")
        return did_accident_occur

    def generate_race_track(self,track_name=TrackName.IstanbulPark):
        road_net_gen = RoadNetworkGenerator()
        way_points = road_net_gen.get_way_points(track_name)
        return way_points
    
    def set_road_network(self,way_points,width_scale=1.0):
        self.client.call("SetRoadNetwork", way_points, True, False, 
                False,width_scale)

    def set_sync_mode(self,is_active,time_dilation=1.0):
        self.client.call("SetSynchronousMode",bool(is_active),float(time_dilation))

    def tick(self):
        self.client.call("Tick")
    
    def blocking_tick(self):
        self.client.call("TickWithWait")

