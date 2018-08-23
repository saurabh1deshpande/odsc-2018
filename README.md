# ODSC India 2018

This is source code repository for session "Introduction to Reinforcement Learning using Python and OpenAI Gym" at ODSC India 2018 conference.


# Requirements
Linux\
Python >3.5\
gym==0.10.5\
numpy==1.14.0\
matplotlib==2.1.2\
torch==0.4.0

# Installation of Prerequisites

<Your Python Environment’s pip path>pip install -r requirements.txt

For Multi Armed Bandits Code, please clone and install environments from following repo,\
sudo git clone https://github.com/JKCooper2/gym-bandits.git\
cd gym-bandits/
<your environment pip path> pip install .

For DQN you will have to install,\
pytorch (https://pytorch.org/)\
opencv (conda install -c conda-forge opencv)\
pip install gym[atari]

If gym[atari] installation errors out on cmake not being installed,\
To install cmake,

sudo apt-get install zlib1g-dev\
sudo apt-get install software-properties-common\
sudo add-apt-repository ppa:george-edison55/cmake-3.x\
sudo apt-get update\
sudo apt-get install cmake

DQN code is copied from https://github.com/Vetal1977/pong-rl. It is divided it into two files,
pong_nn.py - Training and model persistence. It creates directory called 'model' in the same path and saves intermediate and final models.
Trained model is checked under 'model' directory. It took around 5 hours to train on my personal hardware (ubuntu 14.04, 64GB RAM, 1 NVIDEA 1050 2GB) 

To directly run pong game using pre-trained model, run,  pong_nn_play.py