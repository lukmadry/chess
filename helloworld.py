import pybullet as p
import pybullet_data
import time

# 1. Connect to PyBullet in GUI mode (opens a 3D window)
# Use p.DIRECT instead of p.GUI if you ever want to run headless (no window)
physics_client = p.connect(p.GUI)

# 2. Add the official PyBullet URDF assets path (gives access to plane, robots, etc.)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 3. Set world gravity (pointing down along the Z-axis)
p.setGravity(0, 0, -9.81)

# 4. Load a ground plane at position (0, 0, 0)
plane_id = p.loadURDF("plane.urdf")

# 5. Load a sample object 2 metres in the air
# Orientation is set using quaternions (x, y, z, w); [0,0,0,1] is no rotation
start_pos = [0, 0, 2.0]
start_orientation = p.getQuaternionFromEuler([0, 0, 0])
robot_id = p.loadURDF("r2d2.urdf", start_pos, start_orientation)

print(f"Connected to PyBullet. Created Plane ID: {plane_id}, Object ID: {robot_id}")

# 6. Run the simulation loop
try:
    for _ in range(500):
        # Advance the physics simulation by 1 step (1/240th of a second by default)
        p.stepSimulation()
        time.sleep(1.0 / 240.0)  # Pause to match real-time speed

        # Print the object's current height (Z-coordinate)
        pos, orientation = p.getBasePositionAndOrientation(robot_id)
        print(f"Current Z-height: {pos[2]:.2f} m", end="\r")

except KeyboardInterrupt:
    pass

finally:
    # 7. Disconnect and clean up
    p.disconnect()
    print("\nDisconnected cleanly.")