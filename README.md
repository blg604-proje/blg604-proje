# Environment Setup

Download Simstar from the link:

[TO BE UPDATED SOON]

### Windows 
Just click on Simstar.exe and Simstar is ready

### Linux 
    cd Simstar
  
    chmod 777 -R *
  
    ./Simstar.sh

## Install Python API

      cd PythonAPI

      python setup.py install


## Installation Test

There are multiple stages that needs to be checked. 

### 1. Test Simstar Executable

Open the simstar executable, allow for networking if asked. 

![opening_screen](PythonAPI/img/opening_screen.png)

### 2. Test PythonAPI installation

Run the following with success.

	cd PythonAPI

	python python_api_intro.py

### 3. Test Environment Setup

	cd GymEnv

	python example_experiment.py


### Optional Test

To test a closed loop training with Pytorch, you can run the example DDPG agent from examples folder.

