# ECE449 Fall 2025
# Project Controller
#
# Jeremy Carefoot
# Raashid Hamdan
# Zeeshan Haque

from kesslergame import KesslerController
from typing import Dict, Tuple
import skfuzzy as fuzz
import numpy as np
import math
from skfuzzy import control as ctrl

class ProjectController(KesslerController):

    def __init__(self):
        """
        Any variables or initialization desired for the controller can be set up here
        """
        self.setup_mine_control()

    def setup_mine_control(self):
        # a shorthand for asteroid
        ship_thrust = ctrl.Antecedent(np.arange(0, 480, 1), 'ship_thrust')
        closest_a_dist = ctrl.Antecedent(np.arange(0, 1000, 1), 'closest_a_dist')
        drop_mine = ctrl.Consequent(np.arange(-1, 1, 0.1), 'drop_mine')

        # Fuzzy sets for ship thrust
        ship_thrust['S'] = fuzz.zmf(ship_thrust.universe, 0, 50)
        ship_thrust['M'] = fuzz.trimf(ship_thrust.universe, [0, 200, 300])
        ship_thrust['L'] = fuzz.smf(ship_thrust.universe, 200, 480)

        # Fuzzy closest asteroid distance
        closest_a_dist['S'] = fuzz.zmf(closest_a_dist.universe, 0, 150)
        closest_a_dist['M'] = fuzz.trimf(closest_a_dist.universe, [0, 250, 500])
        closest_a_dist['L'] = fuzz.smf(closest_a_dist.universe, 250, 1000)

        # Fuzzy set for consequent (drop mine or not)
        drop_mine['N'] = fuzz.trimf(drop_mine.universe, [-1, -1, 0.0])
        drop_mine['Y'] = fuzz.trimf(drop_mine.universe, [0.0, 1, 1]) 

        # Declare fuzzy rules
        rules = [
            ctrl.Rule(ship_thrust['S'], drop_mine['N']),
            ctrl.Rule(ship_thrust['M'] & (closest_a_dist['L'] | closest_a_dist['M']), drop_mine['N']),
            ctrl.Rule(ship_thrust['L'] & (closest_a_dist['L'] | closest_a_dist['M']), drop_mine['N']),
            ctrl.Rule((ship_thrust['M'] | ship_thrust['L']) & closest_a_dist['S'], drop_mine['Y']),
        ]

        self.mine_control = ctrl.ControlSystem(rules)

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        """
        Method processed each time step by this controller to determine what control actions to take
        """

        ship_x = ship_state["position"][0]     # See src/kesslergame/ship.py in the KesslerGame Github
        ship_y = ship_state["position"][1]       
        closest_asteroid = None

        # Finds closest asteroid
        for a in game_state["asteroids"]:
            # euclidian distance
            curr_dist = math.sqrt((ship_x - a["position"][0])**2 + (ship_y - a["position"][1])**2)

            if closest_asteroid is None :
                closest_asteroid = dict(aster = a, dist = curr_dist)
                
            else:    
                # closest_asteroid exists, and is thus initialized. 
                if closest_asteroid["dist"] > curr_dist:
                    # New minimum found
                    closest_asteroid["aster"] = a
                    closest_asteroid["dist"] = curr_dist

        mine_sys = ctrl.ControlSystemSimulation(self.mine_control,flush_after_run=1)

        # Note: These are the parameters that must be implemented using genetic fuzzy systems
        thrust = 250
        turn_rate = -90
        fire = True

        # Calculate if mine should be dropped
        mine_sys.inputs({
            'closest_a_dist': closest_asteroid['dist'],
            'ship_thrust': abs(thrust)
        })
        mine_sys.compute()

        # Set whether mine is dropped
        drop_mine = True if mine_sys.output['drop_mine'] >= 0 else False

        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        return "Project Controller"