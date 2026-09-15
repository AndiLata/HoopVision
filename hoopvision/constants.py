"""Regulation basketball dimensions.

These are the reason this pipeline can report metres instead of pixels.
Every one of them is a fixed, published number - which makes the scene
self-calibrating in a way most single-camera CV problems are not.
"""

# Rim
RIM_INNER_DIAMETER_M = 0.4572      # 18 in, FIBA/NBA/NCAA/NFHS all agree
RIM_HEIGHT_M = 3.048               # 10 ft, floor to top of ring

# Ball (size 7 - men's regulation)
BALL_DIAMETER_M = 0.2395           # 29.5 in circumference
BALL_RADIUS_M = BALL_DIAMETER_M / 2

# Size 6 (women's regulation), if you switch balls mid-project
BALL_DIAMETER_SIZE6_M = 0.2286

# Backboard
BACKBOARD_WIDTH_M = 1.8288         # 72 in
BACKBOARD_HEIGHT_M = 1.0668        # 42 in

# Physics
G = 9.80665                        # m/s^2

# The horizontal room a ball centre has at the rim plane before the ball
# clips the ring. This is the tolerance the make/miss test uses.
FITS_THROUGH_HALF_WIDTH_M = (RIM_INNER_DIAMETER_M - BALL_DIAMETER_M) / 2  # ~0.1089 m

# Entry-angle window widely cited as optimal for a jump shot.
OPTIMAL_ENTRY_ANGLE_DEG = (43.0, 47.0)
