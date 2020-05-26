# simstar_env

Download Simstar from the link:

https://drive.google.com/open?id=1Gp-XXnOX9dbDcfqFJNJ4UtZqo9sWqjUg

### Windows 
Just click on Simstar.exe and Simstar is ready

### Linux 
    cd Simstar
  
    chmod 777 -R *
  
    ./Simstar.sh

## Requirements

### Python Package Requirements

#### Option 1: Install using Anaconda
Create a new environment using anaconda. 

	conda env create --file environment.yml

	conda activate final604


#### Option 2: Install using pip
	
Install required python libraries from requirements.txt by

	pip install -r requirements.txt


### Pytorch Installation

Follow the official guide from the [link](https://pytorch.org).

The final evaluation will be using pytorch version 1.5 and CUDA version 10.2.


## Install Python API

      cd PythonAPI

      python setup.py install


## Installation Test

There are multiple stages that needs to be checked. 

### 1. Test Simstar Executable

Open the simstar executable, allow for networking if asked. 

![opening_screen](PythonAPI/img/opening.png)

### 2. Test PythonAPI installation

Run the following with success.

	cd PythonAPI

	python python_api_intro.py

### 3. Test Environment Setup

	cd GymEnv

	python example_experiment.py


### Optional Test

To test a closed loop training with Pytorch, you can run the example DDPG agent from examples folder.

	cd examples/pytorch

	python train.py

