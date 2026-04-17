#TODO: Add documentation to message
# following include/RoboCupGameControllData.h's RoboCupGameControlData structure

uint8 team_number
uint8 field_player_colour
uint8 goalkeeper_colour
uint8 goalkeeper
uint8 score
uint8 penalty_shot
uint16 single_shots
uint16 message_budget
RobotInfo[] players         # the team's players (length=11)