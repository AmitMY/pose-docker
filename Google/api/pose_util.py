
#        8   12  16  20
#        |   |   |   |
#        7   11  15  19
#    4   |   |   |   |
#    |   6   10  14  18
#    3   |   |   |   |
#    |   5---9---13--17
#    2    \         /
#     \    \       /
#      1    \     /
#       \    \   /
#        ------0-

# Anatomy guide http://blog.handcare.org/blog/2017/10/26/anatomy-101-finger-joints/
HAND_POINTS = [
    "BASE",
    "T_STT", "T_BCMC", "T_MCP", "T_IP",  # Thumb
    "I_CMC", "I_MCP", "I_PIP", "I_DIP",  # Index
    "M_CMC", "M_MCP", "M_PIP", "M_DIP",  # Middle
    "R_CMC", "R_MCP", "R_PIP", "R_DIP",  # Ring
    "P_CMC", "P_MCP", "P_PIP", "P_DIP",  # Pinky
]


def hand_from_raw(hand):
    return {name:  {"x": hand[i][0], "y": hand[i][1], "z": hand[i][2]} for i, name in enumerate(HAND_POINTS)}