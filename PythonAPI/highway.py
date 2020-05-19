try:
    import numpy as np
except ImportError:
    raise ImportError('pip install numpy')

import simstar
import os 
import argparse
import logging
import time


if __name__ == '__main__':  
    argparser = argparse.ArgumentParser(description='Simstar API')
    argparser.add_argument('--host',metavar='H',default='127.0.0.1',help='IP')
    argparser.add_argument('-p', '--port',metavar='P',default=8080,type=int,help='Port')
    args = argparser.parse_args()
    log_level = logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('connecting to server at %s:%s', args.host, args.port)

    client = simstar.Client(args.host,args.port)
    
    road_location = simstar.WayPoint(0.0, 0.0, 0.0)
    road_rotation = simstar.WayPointOrientation(0.0, 0.0, 0.0)
    
    road_1 = client.create_road_generator(spawn_location = road_location, spawn_rotation = road_rotation, number_of_lanes = 3)
    road_1.add_road(radius = 1000, angle = 0)
    
    road_location = simstar.WayPoint(0.0, 100.0, 0.0)
    
    road_2 = client.create_road_generator(spawn_location = road_location, spawn_rotation = road_rotation, number_of_lanes = 4)
    road_2.add_road(radius = 1000, angle = 0)
    
    trajectory = []
    trajectory.append(simstar.TrajectoryData(0, 0, 0, False, 0))
    
    vehicle_1 = client.spawn_vehicle(actor = road_1, distance = 5, lane_id = 1, initial_speed = 0, 
                set_speed = 0, trajectory_data = [])
    vehicle_2 = client.spawn_vehicle(actor = road_2, distance = 40, lane_id = 4, initial_speed = 0, 
                set_speed = 100, trajectory_data = [])
    vehicle_3 = client.spawn_vehicle(actor = road_2, distance = 10, lane_id = 1, initial_speed = 5, 
                set_speed = 20, trajectory_data = [], vehicle_type = VehicleType.Motorbike_2)
    vehicle_4 = client.spawn_vehicle(actor = vehicle_2, distance = 10, lane_id = 4, initial_speed = 20, 
                set_speed = 30, trajectory_data = [])
    
    client.display_vehicle(vehicle_2)
    vehicle_2.set_controller_type(simstar.DriveType.Keyboard)
    
    vehicle_2.control_vehicle(1.0,1.0,0.0)

    agents = []
    agents.append(vehicle_3)
    agents.append(vehicle_4)
    
    client.autoPilotAgents(agents)
    
    time.sleep(5)
    
    agents = []
    agents.append(vehicle_2)
    client.autoPilotAgents(agents)
    
    client.display_vehicle(vehicle_3)
    vehicle_3.set_controller_type(simstar.DriveType.API)
    vehicle_3.control_vehicle(1.0,-1.0,0.0)
    time.sleep(5)
    
    actors = []
    actors.append(vehicle_2)
    actors.append(vehicle_3)
    actors.append(vehicle_4)
    actors.append(road_2)
    client.removeActors(actors)    
 