import os
import pygame
import cv2
import numpy as np

class Interface:
    """ Pygame interface for training TAMER with OpenCV slider """

    def __init__(self, action_map):
        self.action_map = action_map
        pygame.init()
        self.font = pygame.font.Font("freesansbold.ttf", 40)

        # Create Pygame window for displaying actions
        self.action_screen = pygame.display.set_mode((600, 300))  # Increased size for better visibility
        pygame.display.set_caption("Agent Action Display")

        self.reward_value = 0  # Initial reward value
        self.create_opencv_slider()

        # Initial draw for action screen
        self.draw_action_interface()

    def create_opencv_slider(self):
        """ Create OpenCV slider for reward value """
        cv2.namedWindow("Reward Slider", cv2.WINDOW_NORMAL)  # Create a window for the slider, allow resizing
        cv2.resizeWindow("Reward Slider", 600, 150)  # Resize window to make it larger
        cv2.createTrackbar("Reward", "Reward Slider", 100, 200, self.update_reward)  # Slider range from -1 to 1 (scaled)

    def update_reward(self, value):
        """ Update reward value based on slider position """
        self.reward_value = value / 100 - 1  # Scale to range [-1, 1]

    def draw_action_interface(self):
        """ Draw the action interface in the Pygame window """
        self.action_screen.fill((0, 0, 0))  # Clear the action screen
        text = self.font.render(f"Action: {self.action_map[0]}", True, (255, 255, 255))
        text_rect = text.get_rect(center=(300, 150))  # Centered in the action window
        self.action_screen.blit(text, text_rect)
        pygame.display.update()

    def get_scalar_feedback(self):
        """
        Get human input via the OpenCV slider.
        Returns: scalar reward (range from -1 to 1)
        """
        return self.reward_value

    def show_action(self, action):
        """
        Show agent's action on action window
        Args:
            action: numerical action (for MountainCar environment only currently)
        """
        self.action_screen.fill((0, 0, 0))  # Clear the action screen
        text = self.font.render(self.action_map[action], True, (255, 255, 255))
        text_rect = text.get_rect(center=(300, 150))  # Centered in the action window
        self.action_screen.blit(text, text_rect)
        pygame.display.update()

    def update(self):
        """ Update method to handle events and refresh the Pygame window """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                cv2.destroyAllWindows()  # Close the OpenCV window
                exit()

        self.draw_action_interface()  # Draw action display
        cv2.imshow("Reward Slider", np.zeros((150, 600, 3), dtype=np.uint8))  # Show an empty frame for the slider window
        cv2.waitKey(1)  # Allow OpenCV to process events


# Example usage
if __name__ == "__main__":
    action_map = {
        0: 'Move Left',
        1: 'Move Right',
        2: 'Jump',
        3: 'Crouch'
    }
    interface = Interface(action_map)

    # Test interface loop
    running = True
    while running:
        interface.update()
