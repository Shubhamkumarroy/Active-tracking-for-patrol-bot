import math
class Robot:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Target:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def calculate_distance(robot, target):
    return math.sqrt((robot.x - target.x)**2 + (robot.y - target.y)**2)

def move_towards_target(robot, target, speed):
    distance = calculate_distance(robot, target)
    
    if distance <= speed:
        # Robot can reach the target in one step
        robot.x = target.x
        robot.y = target.y
    else:
        # Calculate the unit vector towards the target
        dx = (target.x - robot.x) / distance
        dy = (target.y - robot.y) / distance
        
        # Move the robot towards the target
        robot.x += dx * speed
        robot.y += dy * speed

# Example usage
if __name__ == "__main__":
    robot = Robot(0, 0)
    target = Target(5, 5)
    speed = 1.0
    
    while calculate_distance(robot, target) > 0:
        move_towards_target(robot, target, speed)
        print(f"Robot Position: ({robot.x}, {robot.y})")

    print("Robot reached the target!")
