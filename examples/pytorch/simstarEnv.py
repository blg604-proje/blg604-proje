
import gym 
from gym import spaces
import collections as col
import numpy as np 
import time
import sys
import os
import pickle

try:
    import simstar
except ImportError:
    print("go to PythonAPI folder where setup.py is located")
    print("python setup.py install")


"""
Parameters Overview:
    track: the type of race track to generate 
    speed_up: how faster should simulation run. up to 10x. down to 0.1x
    synronized_mode: simulator waits for update signal from client if enabled
    width_scale : the scale of the road width
    hz: sampling of the environment in hertz
"""

class SimstarEnv(gym.Env):

    def __init__(self,host="127.0.0.1",port=8080,track=simstar.TrackName.HungaryGrandPrix,
            synronized_mode=False,hz=2,speed_up=1,width_scale=1.5,
            add_agent=False,agent_set_speed=30,agent_rel_pos=50,
            autopilot_agent=True):
        
        self.add_agent = add_agent
        self.agent_set_speed = agent_set_speed
        self.agent_rel_pos = agent_rel_pos
        self.autopilot_agent = autopilot_agent
        self.default_speed = 130
        self.road_width = 4.5 * width_scale
        self.track_sensor_size = 19
        self.opponent_sensor_size = 36
        self.fps = 60
        self.hz = hz
        self.tick_number_to_sample = self.fps//hz
        self.track_name = track 
        self.synronized_mode = synronized_mode
        self.sync_step_num = self.tick_number_to_sample//speed_up
        self.speed_up = speed_up
        self.width_scale = width_scale
        self.client = simstar.Client(host=host,port=port)
        try:
            self.client.ping()
        except:
            print("******* Make sure a Simstar instance is open and running *******")
        

        self.client.reset_level()

        self.client.create_road_generator(number_of_lanes=1)

        self.apply_settings()
        
        # if road network is saved. load. if not, make query and generate
        try:
            with open(str(self.track_name)+".pkl", "rb") as fp:
                track_points = pickle.load(fp)
        except:
            track_points = self.client.generate_race_track(self.track_name)
            with open(str(self.track_name)+".pkl", "wb") as fp: 
                pickle.dump(track_points, fp)
        
        # generate road network
        self.client.set_road_network(track_points,
            width_scale=self.width_scale)


        self.simstar_step(2)

        # a list contaning all vehicles 
        self.actor_list = []

        #input space. 
        high = np.array([np.inf, np.inf,  1., 1.])
        low = np.array([-np.inf, -np.inf, 0., 0.])
        self.observation_space = spaces.Box(low=low, high=high)
        
        # action space: [steer, accel, brake]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,))
        self.default_action = [0.0,1.0,0.0]
        


    def apply_settings(self):
        self.client.set_sync_mode(self.synronized_mode,self.speed_up)

    def reset(self):
        # delete all the actors 
        self.client.remove_actors(self.actor_list)
        self.actor_list.clear()
        self.simstar_step(2)
        print("[SimstarEnv] actors are destroyed")
        time.sleep(0.5)

        # spawn a vehicle
        self.main_vehicle = self.client.spawn_vehicle(distance=150,lane_id=1,initial_speed=0,set_speed=0)

        self.simstar_step(2)
        
        # add all actors to the acor list
        self.actor_list.append(self.main_vehicle)

        # Add an agent in Auto Pilot
        self.agent = self.client.spawn_vehicle(actor= self.main_vehicle,
            distance=self.agent_rel_pos,lane_id=1,
            initial_speed=0,set_speed=self.agent_set_speed)
        
        self.actor_list.append(self.agent)
        if(self.autopilot_agent):
            # drive this agent in auto pilot mode.
            agents = []
            agents.append(self.agent)
            self.client.autopilot_agents(agents)
        else:
            self.agent.set_controller_type(simstar.DriveType.API)


        # set as display vehicle to follow from simstar
        self.client.display_vehicle(self.main_vehicle)

        self.simstar_step(2)
        
        # set drive type as API.
        self.main_vehicle.set_controller_type(simstar.DriveType.API)
        
        self.simstar_step(2)

        # attach appropriate sensors to the vehicle
        track_sensor_settings = simstar.DistanceSensorParameters(enable = True, 
            draw_debug = False,
            add_noise = False, location_x = 0.0, location_y = 0.0,
            location_z = 0.05, yaw_angle = np.pi*10/360, minimum_distance = 0.2,
            maximum_distance = 200.0, fov = 190.0, 
            update_frequency_in_hz = 60.0,
            number_of_returns=self.track_sensor_size,query_type=simstar.QueryType.Static)
        self.track_sensor = self.main_vehicle.add_distance_sensor(track_sensor_settings)

        self.simstar_step(2)
        
        opponent_sensor_settings = simstar.DistanceSensorParameters(enable = True, 
            draw_debug = False,
            add_noise = False, location_x = 2.0, location_y = 0.0,
            location_z = 0.4, yaw_angle = 0, minimum_distance = 0.0,
            maximum_distance = 200.0, fov = 360.0, 
            update_frequency_in_hz = 60.0,
            number_of_returns=self.opponent_sensor_size,query_type=simstar.QueryType.Dynamic)
        self.opponent_sensor = self.main_vehicle.add_distance_sensor(opponent_sensor_settings)

        self.simstar_step(2)
        
        simstar_obs = self.get_simstar_obs(self.default_action)
        observation = self.make_observation(simstar_obs)
        return observation

    def calculate_reward(self,simstar_obs):
        collision = simstar_obs["damage"]
        reward = 0.0
        done = False

        trackPos =  simstar_obs['trackPos']
        spx = simstar_obs['speedX']
        spy = simstar_obs['speedY']
        sp = np.sqrt(spx*spx + spy*spy)
        angle = simstar_obs['angle']

 
        progress = sp *(np.cos(np.abs(angle)) - np.abs(np.sin(np.abs(angle))) - np.abs(trackPos) )

        reward = progress
        
        # for debuggging purposes
        #print("angle: %2.2f,speed %2.2f, trackPos %2.2f"%(angle,sp,trackPos))

        #print("[SimstarEnv] term1 %2.2f, term2 %2.2f, term3 %2.2f, spx %2.2f, spy%2.2f"%\
        #    (np.cos(angle) ,-np.abs(np.sin(angle)), -np.abs(trackPos),spx,spy )   )

        # if collision. finish race
        if(collision):
            print("[SimstarEnv] finish episode bc of Accident")
            reward = -20
            done = True
        
        # if the car has gone off road
        if(abs(trackPos)>1.0):
            print("[SimstarEnv] finish episode bc of road deviation")
            reward = -20
            done = True
        # if the car has returned backward, end race
        if( abs(angle)>(np.pi)/1.8 ):
            print("[SimstarEnv] finish episode bc of going backwards")
            reward = -20
            done = True
        
        # TODO: if vehicle too slow. restart
        
        return reward,done

    def get_agent_obs(self):
        vehicle_state = self.agent.get_vehicle_state_self_frame()
        speed_x_kmh = abs( self.ms_to_kmh( float(vehicle_state['velocity']['X_v']) ))
        speed_y_kmh = abs(self.ms_to_kmh( float(vehicle_state['velocity']['Y_v']) ))
        opponents = self.opponent_sensor.get_sensor_detections()
        track = self.track_sensor.get_sensor_detections()
        road_deviation = self.agent.get_road_deviation_info()
        if not len(track):
            track = self.track_sensor.get_sensor_detections()
            road_deviation = self.agent.get_road_deviation_info()
        
        speed_x_kmh = np.sqrt(speed_x_kmh*speed_x_kmh + speed_y_kmh*speed_y_kmh)
        speed_y_kmh = 0.0
        angle = float(road_deviation['yaw_dev'])
        
        trackPos = float(road_deviation['lat_dev'])/self.road_width

        damage = bool( self.agent.check_for_collision() )

        agent_simstar_obs = {'speedX': speed_x_kmh,
                        'speedY':speed_y_kmh,
                        'opponents':opponents ,
                        'track': track,
                        'angle': angle,
                        'damage':damage,
                        'trackPos': trackPos
                    }
        return self.make_observation(agent_simstar_obs)

    # [steer, accel, brake] input
    def set_agent_action(self,action):
        steer = float(action[0])
        throttle = float(action[1])
        brake = float(action[2])
        steer = steer/4
        brake = brake/8
        if(brake<0.01):
            brake=0.0
        self.agent.control_vehicle(throttle=throttle,
                                    brake=brake,steer=steer)

    def step(self,action):
        simstar_obs = self.get_simstar_obs(action)
        observation = self.make_observation(simstar_obs)
        #currState = np.hstack((observation.angle, observation.track, observation.trackPos,
        #                            observation.speedX, observation.speedY))
        reward,done = self.calculate_reward(simstar_obs)
        summary = {}
        return observation,reward,done,summary

    def make_observation(self,simstar_obs):
        names = ['angle', 'speedX', 'speedY',
                'opponents','track','trackPos']
        Observation = col.namedtuple('Observation', names)

        return Observation(speedX=np.array(simstar_obs['speedX'], dtype=np.float32)/self.default_speed,
                            speedY=np.array(simstar_obs['speedY'], dtype=np.float32)/self.default_speed,
                            angle=np.array(simstar_obs['angle'], dtype=np.float32),
                            trackPos=np.array(simstar_obs['trackPos'], dtype=np.float32),
                            opponents=np.array(simstar_obs['opponents'], dtype=np.float32)/200.,
                            track=np.array(simstar_obs['track'], dtype=np.float32)/200.)
    
    def ms_to_kmh(self,ms):
        return 3.6*ms

    def clear(self):
        self.client.remove_actors(self.actor_list)

    def end(self):
        self.clear()

    def get_agent_obs(self):
        vehicle_state = self.agent.get_vehicle_state_self_frame()
        speed_x_kmh = abs( self.ms_to_kmh( float(vehicle_state['velocity']['X_v']) ))
        speed_y_kmh = abs(self.ms_to_kmh( float(vehicle_state['velocity']['Y_v']) ))
        opponents = self.opponent_sensor.get_sensor_detections()
        track = self.track_sensor.get_sensor_detections()
        road_deviation = self.agent.get_road_deviation_info()
        if len(track) < self.track_sensor_size:
            track = self.track_sensor.get_sensor_detections()
            road_deviation = self.agent.get_road_deviation_info()
        
        speed_x_kmh = np.sqrt(speed_x_kmh*speed_x_kmh + speed_y_kmh*speed_y_kmh)
        speed_y_kmh = 0.0
        angle = float(road_deviation['yaw_dev'])
        
        trackPos = float(road_deviation['lat_dev'])/self.road_width

        damage = bool( self.agent.check_for_collision() )

        agent_simstar_obs = {'speedX': speed_x_kmh,
                        'speedY':speed_y_kmh,
                        'opponents':opponents ,
                        'track': track,
                        'angle': angle,
                        'damage':damage,
                        'trackPos': trackPos
                    }
        return self.make_observation(agent_simstar_obs)

    # [steer, accel, brake] input
    def set_agent_action(self,action):
        steer = float(action[0])
        throttle = float(action[1])
        brake = float(action[2])
        steer = steer/4
        brake = brake/8
        if(brake<0.01):
            brake=0.0
        self.agent.control_vehicle(throttle=throttle,
                                    brake=brake,steer=steer)



    # [steer, accel, brake] input
    def action_to_simstar(self,action):
        steer = float(action[0])
        throttle = float(action[1])
        brake = float(action[2])
        steer = steer/8
        brake = brake/16
        if(brake<0.01):
            brake=0.0
        self.main_vehicle.control_vehicle(throttle=throttle,
                                    brake=brake,steer=steer)
                                

    def simstar_step(self,step_num=None):
        if not step_num: step_num = self.sync_step_num 
        if(self.synronized_mode):
            while True:
                tick_completed = self.client.tick_given_times(step_num)
                time.sleep(0.005)
                if(tick_completed):
                    break
        else:
            pass

    def get_simstar_obs(self,action):

        self.action_to_simstar(action,self.main_vehicle)

        # required to continue simulation in sync mode
        self.simstar_step()

        vehicle_state = self.main_vehicle.get_vehicle_state_self_frame()
        speed_x_kmh = abs( self.ms_to_kmh( float(vehicle_state['velocity']['X_v']) ))
        speed_y_kmh = abs(self.ms_to_kmh( float(vehicle_state['velocity']['Y_v']) ))
        opponents = self.opponent_sensor.get_sensor_detections()
        track = self.track_sensor.get_sensor_detections()
        road_deviation = self.main_vehicle.get_road_deviation_info()
        print("track sensor size:",len(track))
        if len(track) < self.track_sensor_size:
            self.simstar_step(1)
            opponents = self.opponent_sensor.get_sensor_detections()
            track = self.track_sensor.get_sensor_detections()
            print("size problem len:",len(track))
            self.simstar_step(5)
            track = self.track_sensor.get_sensor_detections()
            print("size problem len:",len(track))

        speed_x_kmh = np.sqrt(speed_x_kmh*speed_x_kmh + speed_y_kmh*speed_y_kmh)
        speed_y_kmh = 0.0
        # deviation from road in radians
        angle = float(road_deviation['yaw_dev'])
        
        # deviation from road center in meters
        trackPos = float(road_deviation['lat_dev'])/self.road_width

        # if collision occurs, True. else False
        damage = bool( self.main_vehicle.check_for_collision() )

        simstar_obs = {'speedX': speed_x_kmh,
                        'speedY':speed_y_kmh,
                        'opponents':opponents ,
                        'track': track,
                        'angle': angle,
                        'damage':damage,
                        'trackPos': trackPos
                    }
        return simstar_obs

    def __del__(self):
        # reset sync mod so that user can interact with simstar
        if(self.synronized_mode):
            self.client.set_sync_mode(False)


if __name__ == "__main__":
    sync = True
    time_to_test = 4
    fps = 60
    hz = 2
    speed_up = 2
    env = SimstarEnv(synronized_mode=sync,
        hz=2,speed_up=speed_up)
    env.reset()
    time.sleep(1)
    print("stepping")
    start_time = time.time()
    for i in range(time_to_test*speed_up):
        test_begin = time.time()
        env.step(env.default_action)
        step_time_diff = time.time() - test_begin
        print("env step time",(step_time_diff)*1e3)
        if not sync:
            time_diff = time.time() - start_time
            time.sleep(1/speed_up-step_time_diff)
            if time_diff > time_to_test:
                print("break from loop")
                break
            
    env.reset()

