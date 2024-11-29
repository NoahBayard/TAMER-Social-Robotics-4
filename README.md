# TAMER-Social-Robotics-4

This project explores and extends the use of the TAMER (Training an Agent Manually via Evaluative Reinforcement) framework. In the words of W. Bradley Knox and Peter Stone, TAMER allows a human to train a learningagent to perform a class of complex tasks by giving scalar reward signals (positive or negative) in response to the agent’s observed actions. The TAMER framework is applied to multiple OpenAI Gym environments, including MountainCar, Acrobot, CartPole, and Taxi, each presenting different challenges in state representation, action selection, and reward structure.

This study also includes experiments on the extended versions of the TAMER framework, such as Deep TAMER and TAMER+RL, which combine human feedback with machine learning methods to further enhance the learning process. We also explore different types of interfaces for providing feedback, including keyboard, mouse, and sliders. The aim is to investigate the efficiency and usability of these interfaces, comparing their impact on learning performance and ease of use for the trainer.

# Instructions 

To train the model in a specific environment, run the desired train script. To evaluate the agent's learning process, evaluate the learned models by running the corresponding evaluate script. To change interfaces, change the "from .interface" line in the desired agent script to "from .interface_mouse" or to "from .interface_slide". To save learned models, don't forget to create a folder to contain them (see code for naming convention).
