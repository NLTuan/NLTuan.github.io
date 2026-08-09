Title: Training a Robot Arm to Pick Up My Cubes
Date: 2026-06-29
Category: Technical
Tags: robotics, machine-learning, vision-language-action, lerobot
Author: VRAI Lab, Le Tuan Huy (Tony) Nguyen, Thavin Thanabalasingam, Dante Aglieri, Jacob Huang, Kim Anh Ton, Laura Tarita, Florian Duquerroix.

Summary: Notes and lessons from training a robot arm, covering the setup, data collection process, policy training loop, evaluation, and the practical lessons learned while moving from model code to real robot behavior.

# Training a Robot Arm to Pick Up My Cubes

**Authors:** VRAI Lab, Le Tuan Huy (Tony) Nguyen, Thavin Thanabalasingam, Dante Aglieri, Jacob Huang, Kim Anh Ton, Laura Tarita, Florian Duquerroix.

In recent years, there have been rapid advances in robotic policies, with hundreds of papers published on Vision-Language-Action (VLA) models and World-Action Models (WAMs). These models have demonstrated remarkable dexterity and generalization capabilities in both simulation and real-world embodiments.

At VRAI Lab, we wanted to find out what it would be like to train our own policies on our own datasets using our own real robots. A few years ago, this would have been out of reach for a small team with a small budget because of the cost of hardware and compute. Thanks to the work of the research and open-source communities, the barrier to entry has fallen. Robotic arms can cost as little as CAD 400, open-source libraries facilitate model training and evaluation, and capable models are now accessible to smaller teams.

As a first step, we trained and deployed ACT on an SO-101 robotic arm. We chose a task in which the arm reaches for and grasps a block from an arbitrary starting position, then places it within a defined boundary. Although simple, the task demonstrates the model's ability to complete a real-world task: recognize the block, reach for it, place it in the correct location, and finish gracefully.

We are impressed by the model's performance and are now working to improve its task diversity and generalization, language understanding, and capabilities through a bimanual embodiment.

# Robot hardware
The robot is an [SO-101](https://huggingface.co/docs/lerobot/so101), an affordable CAD 400 robot arm with a gripper and six degrees of freedom (DoF). We chose it because it was affordable, easy to set up, and well supported by the open-source community. Details about the robot parts and 3D-printed components are documented in [its repository](https://github.com/TheRobotStudio/SO-ARM100). Using the robot is mostly smooth sailing; most issues stem from USB-port management for the servo controllers and from cameras randomly disconnecting or changing device assignments. Our single-arm camera setup consists of one wrist view and one top view. Careful camera placement is important: the task must be solvable from the camera views alone, as those are the inputs available to the robot during inference.

# Task Selection and Data Collection
The chosen task is simple: pick up a red block and place it within an area delimited by red tape. It serves as an introductory exercise in robot imitation learning while requiring the robot to use vision from multiple camera views and some dexterity to complete the task. The short trajectories also make it relatively easy to record. We recorded data using the LeRobot platform, which makes it straightforward to capture demonstrations with leader and follower arms. With this setup, we recorded more than 75 episodes in an hour, with each episode taking 25–30 seconds. We aim for at least 120 episodes per dataset.

# Policy training (ACT)
We also trained the policy with the LeRobot framework. It supports many architectures, including the lightweight [Action Chunking Transformer (ACT)](https://arxiv.org/abs/2304.13705). With 53 million parameters, ACT trains quickly and lets us iterate on hyperparameters efficiently. Training for approximately 1.5 hours on two free Kaggle T4 GPUs produced a policy capable of performing the pick-and-place task. Checkpoint selection is difficult because evaluation occurs in the real world. Loss alone is not a useful metric: robot data are multimodal, so many different trajectories can complete the same task. Instead, we evaluate checkpoints at regular intervals by running rollouts and tallying successful attempts. This process is currently time-consuming, so we are exploring other evaluation methods.

# Results
The resulting model exceeded our initial expectations: we obtained a capable policy with little training and tuning, suggesting that ACT is robust for this task. In particular, once the arm grasps the cube, it reliably completes the placement. ACT generates a new action chunk at every time step, making it responsive to changing conditions and well suited to quick error recovery.

<figure>
  <video controls playsinline preload="metadata" style="max-width: 100%; height: auto;">
    <source src="https://github.com/NLTuan/NLTuan.github.io/releases/download/robot-arm-assets/repeat-successes-web.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <figcaption>A montage of successful pick-and-place attempts during evaluation.</figcaption>
</figure>

Of course, the model still has limitations. First, because ACT uses action-chunk ensembling, we had to tune the ensembling factor to obtain smoother movement. Second, the model struggles to complete the task on consecutive attempts. We suspect that ensembling contributes to this issue, as older action chunks can influence later chunks. Third, there are "dead zones" in which the policy cannot complete the task and becomes stuck. More diverse data may address this limitation. Finally, ACT retains a deterministic loss component despite using a CVAE to account for multimodality. We will discuss this further in future work.

# Future work
Our success on this simple task is only a starting point for more capable robotics work. We plan to expand the robot's capabilities with a bimanual setup that can unlock more diverse tasks, such as folding clothes or inserting batteries. We also want to explore multitask vision-language-action settings, where a generalist policy can be prompted to perform a task rather than trained from scratch for a single task, as in our experiment. Finally, the field has shifted towards diffusion- and flow-matching-based approaches to capture the multimodality of robotic data. To address the latter two directions, we are currently fine-tuning SmolVLA, a lightweight 400-million-parameter flow-matching model pretrained on diverse SO-101 robot data.
