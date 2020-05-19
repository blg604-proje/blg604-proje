

import numpy 
import msgpackrpc

from .types import *



class Sensor():
    def __init__(self, client, ID):
        self._ID = int(ID)
        self.client = client

    def getID(self):
        return self._ID
        
    def remove_sensor(self):
        return self.client.call("RemoveSensor", self._ID)
