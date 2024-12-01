# TAMER-Social-Robotics-4

This project explores and extends the use of the TAMER (Training an Agent Manually via Evaluative Reinforcement) framework. In the words of W. Bradley Knox and Peter Stone, TAMER allows a human to train a learning agent to perform a class of complex tasks by giving scalar reward signals (positive or negative) in response to the agent’s observed actions. The TAMER framework is applied to multiple OpenAI Gym environments, including MountainCar, Acrobot, CartPole, and Taxi, each presenting different challenges in state representation, action selection, and reward structure.

This study also includes experiments on the extended versions of the TAMER framework, such as Deep TAMER and TAMER+RL, which combine human feedback with machine learning methods to further enhance the learning process. We also explore different types of interfaces for providing feedback, including keyboard, mouse, and sliders. The aim is to investigate the efficiency and usability of these interfaces, comparing their impact on learning performance and ease of use for the trainer.

## Instructions 

1. **Training**  
   To train the model in a specific environment, run the desired train script.  
   
2. **Evaluation**  
   To evaluate the agent's learning process, evaluate the learned models by running the corresponding evaluate script.  

3. **Interface Customization**  
   To change interfaces, modify the `from .interface` line in the desired agent script to one of the following options:  
   - `from .interface_mouse`  
   - `from .interface_slide`  

4. **Model Saving**  
   To save learned models, create a folder to contain them. Check the code for naming conventions to ensure models are saved correctly.

> **Note**: For Deep TAMER, the `agent_deep_mc` code can be adapted for continuous environments like CartPole and Acrobot by simply modifying the action map in the code.

---

## To-Do / To Be Done

- **Oracle for Human Feedback**  
  Implement an oracle to simulate human feedback during training episodes. This allows:  
  - Conducting many more episodes without requiring manual human feedback.  
  - Randomizing the training and evaluation processes for the Taxi environment.  
  - Applying the Deep TAMER framework effectively to the Taxi environment.  

- **DQN Implementation**  
  Implement the Deep Q-Network (DQN) algorithm for further experimentation. The neural network logic is already implemented within Deep TAMER, making this integration a natural next step.
