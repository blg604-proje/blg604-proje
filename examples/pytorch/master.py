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
NO_WINDOW_OPTION = ' -nullrhi'

def start_sh_script(port=8080,c_port=2000):
    sh_command = 'sh '+  CURRENT_DIR + SIMSTAR_SH_RELATIVE_PATH + \
        NO_WINDOW_OPTION + ' -api-port=' + str(port) + ' -carla-world-port='+str(c_port)
    print("sh command: ",sh_command)
    os.popen(sh_command)

if __name__ == "__main__":
    PARAM_SET = [{'hz':10},{'hz':5}]
    NUM_SIM = 2
    for N in range(NUM_SIM):
        SAVE_NAME= "chekpoint_"+str(N) 
        PORT = 8081+N
        c_port = 2004 + 4*N
        params = PARAM_SET[N]
        hz = params['hz']

        # creating thread 
        simstar_start_thread = threading.Thread(target=start_sh_script,
            kwargs={'port': PORT,'c_port':c_port}) 
        train_thread = threading.Thread(target=train,
            kwargs={'save_name': SAVE_NAME,'port': PORT,'hz':hz}) 
    
        # starting simstar thread 1 
        simstar_start_thread.start()
        
        time.sleep(10) 
        
        # starting train thread 1 
        train_thread.start() 
        

    # wait until train thread is completely executed 
    train_thread.join() 
  
    # threads completely executed 
    print("Done!") 