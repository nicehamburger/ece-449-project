# ECE449 Fall 2025
# Project Controller
#
# Jeremy Carefoot
# Raashid Hamdan
# Zeeshan Haque

from kesslergame import KesslerController
from typing import Dict, Tuple

class ProjectController(KesslerController):

    def __init__(self):
        """
        Any variables or initialization desired for the controller can be set up here
        """
        ...

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        """
        Method processed each time step by this controller to determine what control actions to take
        """

        # Note: These are the parameters that must be implemented using genetic fuzzy systems
        thrust = 500
        turn_rate = -90
        fire = True
        drop_mine = False

        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        return "Project Controller"
