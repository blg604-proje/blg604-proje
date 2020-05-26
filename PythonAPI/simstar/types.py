# -*- coding: utf-8 -*-

class VehicleType():
    Sedan_1 = 0
    Sedan_2 = 1
    Hatchback_1 = 2
    Hatchback_2 = 3
    Hatchback_3 = 4
    SUV_1 = 5
    SUV_2 = 6
    SUV_3 = 7
    Sport = 8
    Pickup = 9
    Track_1 = 10
    Track_2 = 11
    Track_3 = 12
    Track_4 = 13
    Motorbike_1 = 14
    Motorbike_2 = 15
    Motorbike_3 = 16
    

class TrackName():
    DutchGrandPrix ="Dutch"
    HungaryGrandPrix = "Hungary"

class ChangeDirection():
    NoChange = 0
    Left = 1
    Right = 2

class DriveType():
    Auto = 0
    Keyboard = 1
    API = 2
    Matlab = 3
    GYM = 4
    Other = 5

class QueryType():
    All = 0
    Dynamic = 1
    Static = 2

class MsgpackMixin:
    def __repr__(self):
        from pprint import pformat
        return "<" + type(self).__name__ + "> " + pformat(vars(self), indent=4, width=1)

    def to_msgpack(self, *args, **kwargs):
        return self.__dict__

    @classmethod
    def from_msgpack(cls, encoded):
        obj = cls()
        #obj.__dict__ = {k.decode('utf-8'): (from_msgpack(v.__class__, v) if hasattr(v, "__dict__") else v) for k, v in encoded.items()}
        obj.__dict__ = { k : (v if not isinstance(v, dict) else getattr(getattr(obj, k).__class__, "from_msgpack")(v)) for k, v in encoded.items()}
        #return cls(**msgpack.unpack(encoded))
        return obj

class SpawnPoint(MsgpackMixin):
    X = 0.0;
    Y = 0.0;
    Z = 0.0;
    Roll = 0.0;
    Pitch = 0.0;
    Yaw = 0.0;
    is_ego_point = False;
    
    def __init__(self, X = 0.0, Y = 0.0, Z = 0.0, Roll = 0.0, Pitch = 0.0, Yaw = 0.0, is_ego_point = False):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.Roll  = Roll
        self.Pitch = Pitch
        self.Yaw   = Yaw
        self.is_ego_point = is_ego_point
    
class VehicleControl(MsgpackMixin):
    throttle = 0.0
    steer    = 0.0
    brake    = 0.0
    
    def __init__(self, throttle = 0.0, steer = 0.0, brake = 0.0):
        self.throttle = throttle
        self.steer = steer
        self.brake = brake

class WayPoint(MsgpackMixin):
    X = 0.0
    Y = 0.0
    Z = 0.0
    
    def __init__(self, X = 0.0, Y = 0.0, Z = 0.0):
        self.X = X
        self.Y = Y
        self.Z = Z
    
class WayPointOrientation(MsgpackMixin):
    Roll  = 0.0
    Pitch = 0.0
    Yaw   = 0.0
    
    def __init__(self, Roll = 0.0, Pitch = 0.0, Yaw = 0.0):
        self.Roll  = Roll
        self.Pitch = Pitch
        self.Yaw   = Yaw

class RadarDetection(MsgpackMixin):
    X = 0.0
    Y = 0.0
    Z = 0.0
    X_v = 0.0
    Y_v = 0.0
    Z_v = 0.0
    
    def __init__(self, X = 0.0, Y = 0.0, Z = 0.0,X_v=0.0,Y_v=0.0,Z_v=0.0):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.X_v = X_v
        self.Y_v = Y_v
        self.Z_v = Z_v


class VisionDetection(MsgpackMixin):
    X = 0.0
    Y = 0.0
    Z = 0.0
    
    def __init__(self, X = 0.0, Y = 0.0, Z = 0.0):
        self.X = X
        self.Y = Y
        self.Z = Z
        
class WayPointList():
    waypoint_list = []
    
class ActorVelocity():
    actor_velocity = []

class Actor(MsgpackMixin):
    actor_id = 1
    actor_class = 1
    actor_pos = WayPoint()
    def __init__(self, actor_id = 0.0, actor_class = 0.0, actor_pos = 0.0):
        self.actor_id = actor_id
        self.actor_class = actor_class
        self.actor_pos = actor_pos

class ActorSpecifications(MsgpackMixin):
    actor_id = 1
    actor_class = 1
    actor_pos = WayPoint()
    actor_waypoints = WayPointList()
    actor_velocity = ActorVelocity()
    
    def __init__(self, actor_id = 0.0, actor_class = 0.0, actor_pos = 0.0, actor_waypoints = 0.0, actor_velocity = 0.0):
        self.actor_id = actor_id
        self.actor_class = actor_class
        self.actor_pos = actor_pos
        self.actor_waypoints = actor_waypoints
        self.actor_velocity = actor_velocity
        
class RoadSpecifications(MsgpackMixin):
    road_waypoints = WayPointList()
    
    def __init__(self, road_waypoints = 0.0):
        self.road_waypoints = road_waypoints

class RadarSettingRPC(MsgpackMixin):
    enable = True
    range = 100.0
    hor_fov_lower = -30
    hor_fov_upper = 30 
    ver_fov_lower = 0
    ver_fov_upper = 10
    add_noise = False  
    draw_lines    = False
    draw_points    = True
    ignore_ground = True
    sensor_name = "Radar00"
    detector_x = 0 
    detector_y = 0
    detector_z = 0
    detection_prob = 1.0
    num_horizontal_rays = 80
    num_vertical_rays = 20
    radar_res_cm = 180

    def __init__(self, enable=True,range = 50.0, draw_lines = False, draw_points = True, 
        hor_fov_lower = -20, hor_fov_upper = 20,
        ver_fov_lower = 0, ver_fov_upper=20,
        add_noise = False,ignore_ground=True,sensor_name="Radar00",detector_x = 0,
        detector_y=0,detector_z=0,detection_prob=0.99,num_horizontal_rays=120,
        num_vertical_rays=20,radar_res_cm=180):
            self.enable = enable
            self.range = range
            self.hor_fov_lower = hor_fov_lower
            self.hor_fov_upper = hor_fov_upper 
            self.ver_fov_lower = ver_fov_lower
            self.ver_fov_upper = ver_fov_upper
            self.add_noise = add_noise  
            self.draw_lines = draw_lines
            self.draw_points = draw_points
            self.ignore_ground = ignore_ground
            self.sensor_name = sensor_name
            self.detector_x = detector_x 
            self.detector_y = detector_y
            self.detector_z = detector_z
            self.detection_prob = detection_prob
            self.num_horizontal_rays = num_horizontal_rays
            self.num_vertical_rays = num_vertical_rays
            self.radar_res_cm = radar_res_cm  
            
class VisionSettingRPC(MsgpackMixin):
    enable = True
    range = 100.0
    hor_fov_lower = -30
    hor_fov_upper = 30 
    ver_fov_lower = 0
    ver_fov_upper = 10
    add_noise = False  
    draw_lines    = False
    draw_points    = True
    ignore_ground = True
    sensor_name = "Vision00"
    detector_x = 0 
    detector_y = 0
    detector_z = 0
    detection_prob = 1.0
    num_horizontal_rays = 80
    num_vertical_rays = 20
    radar_res_cm = 180

    def __init__(self, enable=True,range = 50.0, draw_lines = False, draw_points = True, 
        hor_fov_lower = -20, hor_fov_upper = 20,
        ver_fov_lower = 0, ver_fov_upper=20,
        add_noise = False,ignore_ground=True,sensor_name="Vision00",detector_x = 0,
        detector_y=0,detector_z=0,detection_prob=0.99,num_horizontal_rays=80,
        num_vertical_rays=20,radar_res_cm=180):
            self.enable = enable
            self.range = range
            self.hor_fov_lower = hor_fov_lower
            self.hor_fov_upper = hor_fov_upper 
            self.ver_fov_lower = ver_fov_lower
            self.ver_fov_upper = ver_fov_upper
            self.add_noise = add_noise  
            self.draw_lines = draw_lines
            self.draw_points = draw_points
            self.ignore_ground = ignore_ground
            self.sensor_name = sensor_name
            self.detector_x = detector_x 
            self.detector_y = detector_y
            self.detector_z = detector_z
            self.detection_prob = detection_prob
            self.num_horizontal_rays = num_horizontal_rays
            self.num_vertical_rays = num_vertical_rays
            self.radar_res_cm = radar_res_cm  
            
class RpcFVector(MsgpackMixin):
    X = 0.0
    Y = 0.0
    Z = 0.0
    def __init__(self,X=0.0,Y=0.0,Z=0.0):
        self.X = X
        self.Y = Y
        self.Z = Z

class LaneData(MsgpackMixin):
    Coordinates = RpcFVector()
    Curvature = [0.0]
    CurvatureDerivative = [0.0]
    HeadingAngle = 0.0
    LateralOffset = 0.0
    Strength = 0.0
    Width = 0.0
    Length = 0.0
    def __init__(self,Coordinates=0.0 , Curvature=0.0, CurvatureDerivative=0.0, HeadingAngle=0.0, LateralOffset=0.0, Strength=0.0, Width=0.0, Length=0.0):
            self.Coordinates = Coordinates
            self.Curvature =  Curvature
            self.CurvatureDerivative =  CurvatureDerivative
            self.HeadingAngle = HeadingAngle
            self.LateralOffset = LateralOffset
            self.Strength = Strength
            self.Width = Width
            self.Length = Length

class ControllerTestParamsRPC(MsgpackMixin):
    let_ego_control_steer = False
    let_ego_control_speed = False
    def __init__(self,let_ego_control_steer=False,let_ego_control_speed=False):
        self.let_ego_control_steer = let_ego_control_steer
        self.let_ego_control_speed = let_ego_control_speed


class SpawnPointAPI(MsgpackMixin):
    X = 0.0
    Y = 0.0
    Z = 35060.0
    roll = 0.0
    pitch = 0.0
    yaw = 0.0
    is_ego = 0.0
    init_speed = 50.0
    set_speed = 50.0
    nametag = "none"
    def __init__(self,X=0.0,Y=0.0,Z=3060.0004883,
        roll=0.0,pitch=0.0,yaw=0.0,
        is_ego = False,init_speed = 50.0, set_speed = 50.0,nametag="none"):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.is_ego = is_ego
        self.init_speed = init_speed
        self.set_speed = set_speed
        self.nametag = nametag 
        
class TrajectoryData(MsgpackMixin):
    set_speed_in_kmh = 0.0
    duration_in_seconds = 0.00001
    duration_in_meters = 0.0
    is_lane_change_allowed = False
    lane_change_direction = 0
    def __init__(self, set_speed_in_kmh = 0.0, duration_in_seconds = 0.0, duration_in_meters = 0.0, is_lane_change_allowed = False, lane_change_direction = 0):
        self.set_speed_in_kmh = set_speed_in_kmh
        self.duration_in_seconds = duration_in_seconds
        self.duration_in_meters = duration_in_meters
        self.is_lane_change_allowed = is_lane_change_allowed
        self.lane_change_direction = lane_change_direction  

class DistanceSensorParameters(MsgpackMixin):
    enable = True
    draw_debug = True
    add_noise = True
    location_x = 0.0
    location_y = 0.0
    location_z = 0.0
    yaw_angle = 0.0
    minimum_distance = 0.0
    maximum_distance = 0.0
    fov = 0.0
    update_latency_in_seconds = 0.0
    update_frequency_in_hz = 0.0
    unnorrelated_noise_sigma = 0.0
    number_of_returns = 1
    query_type = QueryType.All
    def __init__(self, enable = True, draw_debug = True, add_noise = True,
     location_x = 0.0, location_y = 0.0, location_z = 0.0, 
     yaw_angle = 0.0, minimum_distance = 0.0, maximum_distance = 0.0,
      fov = 0.0, update_latency_in_seconds = 0.0, 
      update_frequency_in_hz = 0.0, unnorrelated_noise_sigma = 0.0,
      number_of_returns=1,query_type=QueryType.All ):
        self.enable = enable
        self.draw_debug = draw_debug
        self.add_noise = add_noise
        self.location_x = location_x
        self.location_y = location_y
        self.location_z = location_z
        self.yaw_angle = yaw_angle
        self.minimum_distance = minimum_distance
        self.maximum_distance = maximum_distance
        self.fov = fov
        self.update_latency_in_seconds = update_latency_in_seconds
        self.update_frequency_in_hz = update_frequency_in_hz
        self.unnorrelated_noise_sigma = unnorrelated_noise_sigma
        self.number_of_returns = number_of_returns
        self.query_type = query_type
        
