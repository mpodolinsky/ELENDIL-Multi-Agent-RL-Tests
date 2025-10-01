'''This file defines the basic environment that agents will be trained on.'''

# Imports

# Number of agents - 2
#   Agent A - ground asset
#   Agent B - air asset 

# Action space
#   Agent A
#       Movement - [L,R,U,D] (size 1 step)
#   Agent B
#       Movement - [L,R,U,D,H+,H-] (step size proportional to it's height state)
#       Communication to agent
#           [NaN, NaN] if target, agent, or both are outside it's observability regio
#           [x_target,y_target] (referece frame is arbitrary since B has full knowledge of )

# Observation space
#   Agent A 
#       Absolute location within the grid OR agent B knows it's location ?
#       Partial observability of square (configurable size) centered on itself.
#           Within this region 0/1 if cell contains an obstacle
#           Within this region 0/1 if it contains a target
#   Agent B
#       Absolute location within the grid 
#       Location of B with respect to itself (effectively knowing the absolute position of A)
#       It's own height [h-1,h0,h1]
#       Partial observability of square (proportional to it's height) centered on itself.
#           Within this region 0/1 if cell contains an obstacle
#           Within this region 0/1 if it contains a target

# Environment structure
#   Three sizes
#       Small - 10*10
#       Medium - 10*15
#       Large - 20*20
#   Obstacles and targets
#       Randomly generated n obstacles.
#           n is the smallest integer such that n/size**2 > 0.1
#           There must be a possible path to the target
#           The path to the target must not be straight (TBD)
#       1 randomly generated target Agent A must reach

# Rewards (might need some scaling)
#   Global: 
#   Individual: 
#       Agent A: +1 if it reaches the goal
#       Agent B: +0.1 if A moves closer to the target (0 if tangential -0.1 if moves away?)
#   Secondary rewards/punishments: -0.01 for each timestep

# Other environemnt options
# Modifying MPE2 environments (no partial observability), Mustafa might also know something