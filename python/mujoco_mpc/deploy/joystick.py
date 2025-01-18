import pygame
import time

def initialize_joystick():
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("No joystick detected!")
        pygame.quit()
        exit()

    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Joystick initialized: {joystick.get_name()}")
    return joystick

def main():
    joystick = initialize_joystick()

    try:
        while True:
            pygame.event.pump()  # Process events to update joystick state

            # Reading the LT button (axis 2 for most Xbox controllers)
            lt_value = joystick.get_axis(2)

            print(f"LT Value: {lt_value:.3f}")

            time.sleep(1 / 50.0)  # Maintain 50Hz update rate

    except KeyboardInterrupt:
        print("Exiting...")

    finally:
        pygame.quit()

if __name__ == "__main__":
    main()
