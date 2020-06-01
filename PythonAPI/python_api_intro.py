import numpy as np

try:
    import simstar
except ImportError:
    print("go to PythonAPI folder where setup.py is located")
    print("python setup.py install")



import os 
import argparse
import logging
import time

if __name__ == '__main__':  
    argparser = argparse.ArgumentParser(
        description='Simstar API')
    argparser.add_argument('--host',metavar='H',default='127.0.0.1',help='IP')
    argparser.add_argument('-p', '--port',metavar='P',default=8080,type=int,help='Port')
    args = argparser.parse_args()
    log_level = logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)
    
    
    client = simstar.Client(args.host,args.port)
    client.reset_level()
    
    road_gen = client.create_road_generator(number_of_lanes=1)
    
    road_gen.add_road()
    road_gen.add_road()

    logging.info('spawn 3 vehicles')
    vehicle_1 = client.spawn_vehicle(distance = 80,
     lane_id = 1, initial_speed = 0, set_speed = 0)
    
    vehicle_2 = client.spawn_vehicle(distance = 120,
     lane_id = 1, initial_speed = 0, set_speed = 0)

    vehicle_3 = client.spawn_vehicle(distance = 50,
     lane_id = 1, initial_speed = 0, set_speed = 0)


    client.display_vehicle(vehicle_1)

    logging.info('set control type to API')
    vehicle_1.set_controller_type(simstar.DriveType.Keyboard)
    
    vehicle_1.get_road_deviation_info()
    time.sleep(1)
    print(vehicle_1.get_road_deviation_info())


    track_sensor_settings = simstar.DistanceSensorParameters(enable = True, 
            draw_debug = True,
            add_noise = False, location_x = 0.0, location_y = 0.0,
            location_z = 0.05, yaw_angle = 0, minimum_distance = 0.2,
            maximum_distance = 200.0, fov = 190.0, 
            update_frequency_in_hz = 60.0,
            number_of_returns=19,query_type=simstar.QueryType.Static)
    track_sensor = vehicle_1.add_distance_sensor(track_sensor_settings)
    
    for i in range(4):
        dets = track_sensor.get_sensor_detections()
        print(dets)
        time.sleep(1)

    logging.info("remove actors")
    actor_list = []
    actor_list.append(vehicle_1)
    actor_list.append(vehicle_2)
    actor_list.append(vehicle_3)
    client.remove_actors(actor_list)
    
    logging.info("PythonAPI example finished correctly")
    
