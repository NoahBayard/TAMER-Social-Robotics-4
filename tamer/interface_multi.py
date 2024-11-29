import os
import pygame


class Interface:
    """ Pygame interface for training TAMER with mouse clicks for rewards """

    def __init__(self, action_map):
        self.action_map = action_map
        pygame.init()
        self.font = pygame.font.Font("freesansbold.ttf", 32)

        # Set position of pygame window (so it doesn't overlap with gym)
        os.environ["SDL_VIDEO_WINDOW_POS"] = "1000,100"
        os.environ["SDL_VIDEO_CENTERED"] = "0"

        self.screen = pygame.display.set_mode((400, 300))
        self.bright_green_box = pygame.Rect(50, 50, 100, 100)  # Bright green box for +1
        self.light_green_box = pygame.Rect(50, 200, 100, 100)  # Light green box for +0.5
        self.light_red_box = pygame.Rect(250, 200, 100, 100)   # Light red box for -0.5
        self.bright_red_box = pygame.Rect(250, 50, 100, 100)   # Bright red box for -1

        self.draw_boxes()  # Draw boxes initially

    def draw_boxes(self):
        """ Draw the reward boxes """
        self.screen.fill((0, 0, 0))  # Clear the screen
        pygame.draw.rect(self.screen, (0, 255, 0), self.bright_green_box)  # Bright green box for +1
        pygame.draw.rect(self.screen, (200,255,179), self.light_green_box)   # Light green box for +0.5
        pygame.draw.rect(self.screen, (255,107,94), self.light_red_box)     # Light red box for -0.5
        pygame.draw.rect(self.screen, (255, 0, 0), self.bright_red_box)    # Bright red box for -1
        pygame.display.update()  # Update the display

    def get_scalar_feedback(self):
        """
        Get human input via mouse clicks.
        Returns: scalar reward (+1, +0.5, -1, -0.5)
        """
        reward = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.MOUSEBUTTONDOWN:  # Check for mouse button down event
                pos = pygame.mouse.get_pos()  # Get the mouse position
                if self.bright_green_box.collidepoint(pos):  # If clicked on bright green box
                    reward = 1
                elif self.light_green_box.collidepoint(pos):  # If clicked on light green box
                    reward = 0.5
                elif self.bright_red_box.collidepoint(pos):  # If clicked on bright red box
                    reward = -1
                elif self.light_red_box.collidepoint(pos):  # If clicked on light red box
                    reward = -0.5
                self.draw_boxes()  # Redraw boxes to show feedback
                break  # Exit the loop after processing

        return reward  # Return the reward based on the user's action

    def show_action(self, action):
        """
        Show agent's action on pygame screen
        Args:
            action: numerical action (for MountainCar environment only currently)
        """
        self.draw_boxes()  # Ensure boxes are drawn before showing the action
        text = self.font.render(self.action_map[action], True, (255, 255, 255))
        text_rect = text.get_rect(center=(200, 150))  # Centered below the boxes
        area = self.screen.blit(text, text_rect)
        pygame.display.update(area)  # Update the display with the action text
