Title: Training a Robot Arm to Pick up my Cubes
Date: 2026-06-29
Category: Technical
Tags: robotics, machine-learning, vision-language-action, lerobot
Author: VRAI Lab, Le Tuan Huy (Tony) Nguyen, Thavin Thanabalasingam, 
Summary: Notes and lessons from training a robot arm, covering the setup, data collection process, policy training loop, evaluation, and the practical lessons learned while moving from model code to real robot behavior.

# Training a Robot Arm

In recent years, there has been accelerating advances on robotic policies with hundreds of papers being published on Vision-Language-Action (VLA) models and World-Action models (WAMs) with anazing dexterity and generalization capabilities when deployed in simulation and in real-life embodiments.

At VRAI Lab, we wanted to know what it would be like to train our own policies on our own datasets on our own real robots. A few years ago, this goal would have proven to be impossible for our calibre (we are a small team with a small budget!) due to the demands for expensive hardware and compute. Thanks to the generous work from the research and open-source community, the bar of entry has been lowered. Robotic arms can be as cheap as 400 CAD, open-source libraries facilitate model training and evaluation, and research allowed for capable models to be used by smaller teams. 

As our first stepping stone, we wanted to train and deploy ACT on a SO-101 robotic arm. The task that we have chosen is reaching and grasping a block with an arbitrary starting position, then placing it within a defined boundary. The reason why we proceeded with this task is for its simplicity, yet it is still demonstrating the model's ability to accomplish a real-life task: recognizing the block, reaching it, placing it in the right spot, finishing gracefully. 

We are impressed at the model's performance, and are now working on improving it in terms of task diversity and generalization, language understanding, and a bimanual embodiment for extra (explained further in Future Work).


body WIP

Robot hardware
Software stack (lerobot)
Data collection
Policy training (ACT)
Future work