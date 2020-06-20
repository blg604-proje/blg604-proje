import os
import threading 
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

os.chdir(CURRENT_DIR)

"""
This script assumes that you have the following structure

    train.py
    master.py
    SimStar
    --SimStar.sh
"""

from train import train



SIMSTAR_SH_RELATIVE_PATH = '/SimStar/SimStar.sh'


def start_sh_script():
    sh_command = 'sh '+  CURRENT_DIR+SIMSTAR_SH_RELATIVE_PATH
    os.popen(sh_command)

if __name__ == "__main__": 
    # creating thread 
    simstar_start_thread = threading.Thread(target=start_sh_script) 
    train_thread = threading.Thread(target=train) 
  
    # starting thread 1 
    simstar_start_thread.start()
    
    time.sleep(10) 
    
    # starting thread 2 
    train_thread.start() 
  

    # wait until thread 2 is completely executed 
    train_thread.join() 
  
    # both threads completely executed 
    print("Done!") 