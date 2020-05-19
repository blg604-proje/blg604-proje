

import numpy 
import msgpackrpc

from .types import *



class RoadGenerator():
    def __init__(self,client,ID):
        self._ID = int(ID)
        self.client = client

    def getID(self):
        return self._ID
    
    def add_road(self,radius=100,angle=0):
        self.client.call("AddRoadTo",self._ID,radius,angle)
    