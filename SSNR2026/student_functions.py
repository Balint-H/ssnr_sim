# In this script you should implement your logic for controlling the planar arm. These functions are then imported
# in the main task scripts

import numpy as np

def feedback_control(error_in_position, error_in_velocity):
  # This function should return forces that help reduce the error in joint angles and in angular velocity.
  # The signs of forces, accelerations velocities and joint angles are aligned: A positive force gives a positive
  # acceleration. A positive acceleration will increase velocity over time. A positive velocity increases position
  # (angles in this case) over time. Let's define error in terms target-current; a positive error means you need to
  # increase position/velocity.
  # You might want to clip forces to reasonable values as well

  return np.zeros_like(error_in_velocity)


