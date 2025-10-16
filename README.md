# Curtain Task with Spot Robot

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.2.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)

# Overview

This project demonstrates teleoperation of the Boston Dynamics Spot robot for a curtain manipulation task in IsaacSim 4.2.0.

The Spot robot’s arm and body can be controlled in SE(3) space using keyboard input to collect interaction data with the curtain. 

# Requirements

IsaacSim 4.2.0 

# Teleoperation Data Collection
Run the keyboard teleoperation script to control the Spot robot and collect data:

/path-to-isaacsim/isaac-sim-4.2.0/python.sh /path-to-code/script/teleop_se3_agent.py 
