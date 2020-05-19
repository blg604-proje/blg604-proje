try:
    import msgpackrpc
except ImportError:
    raise ImportError('pip install msgpack-rpc-python')

try:
    import numpy as np
except ImportError:
    raise ImportError('pip install numpy')

import simstar
import os 
import argparse
import logging
import time

from simstar.types import *

def change_directory_to_file_location():
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

change_directory_to_file_location()

if __name__ == '__main__':  
    argparser = argparse.ArgumentParser(description='Simstar API')
    argparser.add_argument('--host',metavar='H',default='127.0.0.1',help='IP')
    argparser.add_argument('-p', '--port',metavar='P',default=8080,type=int,help='Port')
    args = argparser.parse_args()
    log_level = logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('connecting to server at %s:%s', args.host, args.port)

    client = simstar.Client(args.host,args.port)
    
    road_location = WayPoint(0.0, 0.0, 0.0)
    road_rotation = WayPointOrientation(0.0, 0.0, 0.0)
    
    road = client.create_road_generator(spawn_location = road_location, spawn_rotation = road_rotation, number_of_lanes = 3)
    road.add_road(radius = 1000, angle = 0)
    
    trajectory = []
    trajectory.append(TrajectoryData(0, 0, 0, False, 0))
    
    vehicle_1 = client.spawn_vehicle(actor = road, distance = 5, lane_id = 1, initial_speed = 0, set_speed = 0, trajectory_data = [])
    vehicle_2 = client.spawn_vehicle(actor = road, distance = 40, lane_id = 3, initial_speed = 0, set_speed = 100, trajectory_data = [])
    
    client.display_vehicle(vehicle_1)
    
    distance_sensor_settings = DistanceSensorParameters(enable = True, draw_debug = True,
	 add_noise = False, location_x = 0.0, location_y = 0.0,
	  location_z = 1.1, yaw_angle = 0, minimum_distance = 0.2,
	   maximum_distance = 200.0, fov = 180.0, update_latency_in_seconds = 0.0, 
	   update_frequency_in_hz = 100.0, unnorrelated_noise_sigma = 0.2,
	   number_of_returns=19,query_type=QueryType.Static)
    
    distance_sensor_settings_2 = DistanceSensorParameters(enable = True, draw_debug = True,
	 add_noise = False, location_x = -2.0, location_y = 0.8,
	  location_z = 0.5, yaw_angle = 2.356, minimum_distance = 0.2,
	   maximum_distance = 10.0, fov = 45.0, update_latency_in_seconds = 0.0, 
	   update_frequency_in_hz = 100.0, unnorrelated_noise_sigma = 0.2,
	   number_of_returns = 1,query_type=QueryType.All)
       
    sensor_1 = vehicle_1.add_distance_sensor(distance_sensor_settings)
    sensor_2 = vehicle_2.add_distance_sensor(distance_sensor_settings_2)
    
    time.sleep(0.1)
    print(sensor_1.get_sensor_detections())
    print(sensor_2.get_sensor_detections())
    
    sensor_1.remove_sensor()
    sensor_2.remove_sensor()
    
    radar_settings = RadarSettingRPC(enable=True,range=150,draw_lines = False,ignore_ground=False,draw_points=True,\
	hor_fov_lower=-45,hor_fov_upper=45,ver_fov_lower=0,ver_fov_upper=10,add_noise=True,radar_res_cm=250,\
	num_horizontal_rays=120)
    
    radar_settings_2 = RadarSettingRPC(enable=True,range=150,draw_lines = False,ignore_ground=False,draw_points=True,\
	hor_fov_lower=-45,hor_fov_upper=45,ver_fov_lower=0,ver_fov_upper=10,add_noise=True,radar_res_cm=250,\
	num_horizontal_rays=200)

    radar_1 = vehicle_1.add_radar_sensor(radar_settings)
    radar_2 = vehicle_2.add_radar_sensor(radar_settings_2)

    time.sleep(0.1)
    print(radar_1.get_sensor_detections())
    print(radar_2.get_sensor_detections())
    
    time.sleep(3)
    
    radar_1.remove_sensor()
    radar_2.remove_sensor()
    
    imu_1 = vehicle_1.add_imu_sensor()
    imu_2 = vehicle_2.add_imu_sensor()

    time.sleep(0.1)
    print(imu_1.get_sensor_detections())
    print(imu_2.get_sensor_detections())
    
    imu_1.remove_sensor()
    imu_2.remove_sensor()
    
    vision_settings = RadarSettingRPC(range=100,draw_lines = False,draw_points=True,
	hor_fov_lower=-40,hor_fov_upper=40,ver_fov_lower=-15,ver_fov_upper=30,
	num_horizontal_rays=100,num_vertical_rays=50)
    
    smart_1 = vehicle_1.add_smart_vision_sensor(vision_settings)
    smart_2 = vehicle_2.add_smart_vision_sensor(vision_settings)

    time.sleep(0.1)
    print(smart_1.get_sensor_detections())
    print(smart_2.get_sensor_detections())
    print(smart_1.get_lane_points())
    print(smart_2.get_lane_points())
    
    time.sleep(3)
    smart_1.remove_sensor()
    smart_2.remove_sensor()
    
    
 