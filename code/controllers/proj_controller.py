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
        self.max_bullet_count = None

        self.setup_mine_control()
        self.setup_fire_control()
        self.setup_thrust_control()

    # ---------------------- MINE CONTROL ---------------------- #
    def setup_mine_control(self):
        ship_thrust = ctrl.Antecedent(np.arange(0, 480, 1), 'ship_thrust')
        mass_density = ctrl.Antecedent(np.arange(0, 0.1, 0.001), 'mass_density')
        drop_mine = ctrl.Consequent(np.arange(-1, 1, 0.1), 'drop_mine')

        # Fuzzy sets for ship thrust
        ship_thrust['S'] = fuzz.zmf(ship_thrust.universe, 0, 50)
        ship_thrust['M'] = fuzz.trimf(ship_thrust.universe, [0, 200, 300])
        ship_thrust['L'] = fuzz.smf(ship_thrust.universe, 200, 480)

        # Fuzzy sets for mass density
        mass_density['S'] = fuzz.zmf(mass_density.universe, 0, 0.01)
        mass_density['M'] = fuzz.trimf(mass_density.universe, [0, 0.02, 0.04])
        mass_density['L'] = fuzz.smf(mass_density.universe, 0.03, 0.1)

        # Fuzzy set for consequent (drop mine or not)
        drop_mine['N'] = fuzz.trimf(drop_mine.universe, [-1, -1, 0.0])
        drop_mine['Y'] = fuzz.trimf(drop_mine.universe, [0.0, 1, 1])

        # Declare fuzzy rules
        rules = [
            ctrl.Rule(ship_thrust['S'], drop_mine['N']),
            ctrl.Rule(ship_thrust['M'] & (mass_density['S'] | mass_density['M']), drop_mine['N']),
            ctrl.Rule(ship_thrust['L'] & (mass_density['S'] | mass_density['M']), drop_mine['N']),
            ctrl.Rule((ship_thrust['M'] | ship_thrust['L']) & mass_density['L'], drop_mine['Y']),
        ]

        self.mine_control = ctrl.ControlSystem(rules)

    # ---------------------- FIRE CONTROL ---------------------- #
    def setup_fire_control(self):
        ammo = ctrl.Antecedent(np.arange(0, 1, 0.01), 'ammo')
        fire_gun = ctrl.Consequent(np.arange(-1, 1, 0.1), 'fire_gun')
        closest_a_dist = ctrl.Antecedent(np.arange(0, 1000, 1), 'closest_a_dist')

        # Fuzzy sets for ammo
        ammo['L'] = fuzz.zmf(ammo.universe, 0, 0.2)
        ammo['M'] = fuzz.trimf(ammo.universe, [0, 0.4, 0.7])
        ammo['H'] = fuzz.smf(ammo.universe, 0.4, 1.0)

        # Fuzzy sets for closest asteroid distance
        closest_a_dist['S'] = fuzz.zmf(closest_a_dist.universe, 0, 150)
        closest_a_dist['M'] = fuzz.trimf(closest_a_dist.universe, [0, 250, 500])
        closest_a_dist['L'] = fuzz.smf(closest_a_dist.universe, 250, 1000)

        # Fuzzy set for consequent (fire gun or not)
        fire_gun['N'] = fuzz.trimf(fire_gun.universe, [-1, -1, 0.0])
        fire_gun['Y'] = fuzz.trimf(fire_gun.universe, [0.0, 1, 1])

        # Declare fuzzy rules
        rules = [
            ctrl.Rule(ammo['H'] | ammo['M'], fire_gun['Y']),
            ctrl.Rule(ammo['L'] & (closest_a_dist['L'] | closest_a_dist['M']), fire_gun['N']),
            ctrl.Rule(ammo['L'] & closest_a_dist['S'], fire_gun['Y'])
        ]

        self.fire_control = ctrl.ControlSystem(rules)

    # ---------------------- THRUST CONTROL ---------------------- #
    def setup_thrust_control(self):
        # Fuzzy input variables
        distance = ctrl.Antecedent(np.arange(0, 1001, 1), 'distance')
        vert_offset = ctrl.Antecedent(np.arange(-500, 501, 1), 'vert_offset')

        # Fuzzy output variable
        thrust = ctrl.Consequent(np.arange(-480, 481, 1), 'thrust')

        # Distance membership functions
        distance['near'] = fuzz.trimf(distance.universe, [0, 0, 300])
        distance['mid'] = fuzz.trimf(distance.universe, [100, 500, 800])
        distance['far'] = fuzz.trimf(distance.universe, [500, 1000, 1000])

        # Vertical offset membership functions
        vert_offset['below'] = fuzz.trimf(vert_offset.universe, [-500, -500, 0])
        vert_offset['center'] = fuzz.trimf(vert_offset.universe, [-50, 0, 50])
        vert_offset['above'] = fuzz.trimf(vert_offset.universe, [0, 500, 500])

        # Thrust membership functions
        thrust['high_up'] = fuzz.trimf(thrust.universe, [150, 480, 480])
        thrust['medium_up'] = fuzz.trimf(thrust.universe, [50, 200, 350])
        thrust['none'] = fuzz.trimf(thrust.universe, [-50, 0, 50])
        thrust['medium_down'] = fuzz.trimf(thrust.universe, [-350, -200, -50])
        thrust['high_down'] = fuzz.trimf(thrust.universe, [-480, -480, -150])

        thrust_rules = [
            ctrl.Rule(distance['near'] & vert_offset['above'], thrust['high_down']),
            ctrl.Rule(distance['near'] & vert_offset['below'], thrust['high_up']),
            ctrl.Rule(distance['mid'] & vert_offset['above'], thrust['medium_down']),
            ctrl.Rule(distance['mid'] & vert_offset['below'], thrust['medium_up']),
            ctrl.Rule(distance['far'], thrust['none']),
            ctrl.Rule(vert_offset['center'], thrust['none'])
        ]

        self.thrust_control = ctrl.ControlSystem(thrust_rules)

    # ---------------------- ASTEROID DATA ---------------------- #
    def collect_asteroid_data(self, asteroids: list, ship_x: float, ship_y: float) -> dict:
        """
        Collects asteroid data in one frame of the game.
        Data includes local mass density to the ship and closest asteroid data.
        """
        DENSITY_RADIUS = 150
        closest_asteroid = None
        total_mass = 0

        # Find closest asteroid and compute local mass density
        for a in asteroids:
            ax, ay = a["position"]
            curr_dist = math.sqrt((ship_x - ax) ** 2 + (ship_y - ay) ** 2)

            if closest_asteroid is None:
                closest_asteroid = dict(aster=a, dist=curr_dist)

            if curr_dist < DENSITY_RADIUS:
                total_mass += a["mass"]

            if curr_dist < closest_asteroid["dist"]:
                closest_asteroid["aster"] = a
                closest_asteroid["dist"] = curr_dist

        density_area = math.pi * (DENSITY_RADIUS ** 2)
        density = total_mass / density_area if density_area > 0 else 0.0

        return {
            "closest_asteroid": closest_asteroid,
            "mass_density": density
        }

    # ---------------------- MAIN ACTIONS ---------------------- #
    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        """
        Called each time step to determine thrust, turning, firing, and mine dropping.
        """
        ship_x, ship_y = ship_state["position"]

        # ---------- Thrust: find nearest asteroid & vertical offset ---------- #
        nearest_dist = float('inf')
        vert_offset = 0.0

        for ast in game_state["asteroids"]:
            ax, ay = ast["position"]
            dist = math.hypot(ax - ship_x, ay - ship_y)
            if dist < nearest_dist:
                nearest_dist = dist
                vert_offset = ay - ship_y

        # Compute thrust using fuzzy thrust controller
        if nearest_dist == float('inf') or len(game_state["asteroids"]) == 0:
            thrust = 0.0
        else:
            thrust_sim = ctrl.ControlSystemSimulation(self.thrust_control, flush_after_run=1)
            thrust_sim.inputs({
                'distance': nearest_dist,
                'vert_offset': vert_offset
            })
            thrust_sim.compute()
            thrust = float(thrust_sim.output['thrust'])

        # ---------- Mine & Fire: use density + ammo ---------- #
        current_ammo = ship_state["bullets_remaining"]

        # Initialize max bullet count if not set (for calculating ammo ratio)
        if self.max_bullet_count is None:
            # Avoid division by zero later
            self.max_bullet_count = current_ammo if current_ammo > 0 else 1

        asteroid_data = self.collect_asteroid_data(game_state["asteroids"], ship_x, ship_y)
        closest_asteroid = asteroid_data["closest_asteroid"]
        mass_density = asteroid_data["mass_density"]

        # Ammo ratio: if infinite ammo (-1), treat as full (1.0)
        if current_ammo == -1:
            ammo_ratio = 1.0
        else:
            ammo_ratio = current_ammo / self.max_bullet_count if self.max_bullet_count > 0 else 0.0

        mine_sys = ctrl.ControlSystemSimulation(self.mine_control, flush_after_run=1)
        fire_sys = ctrl.ControlSystemSimulation(self.fire_control, flush_after_run=1)

        # Calculate if mine should be dropped
        mine_sys.inputs({
            'mass_density': mass_density,
            'ship_thrust': abs(thrust)
        })
        mine_sys.compute()

        # Distance for firing fuzzy system (if no asteroid, treat as far)
        if closest_asteroid is not None:
            closest_dist = closest_asteroid['dist']
        else:
            closest_dist = 1000.0  # effectively "far"

        # Calculate if gun should be fired
        fire_sys.inputs({
            'closest_a_dist': closest_dist,
            'ammo': ammo_ratio
        })
        fire_sys.compute()

        drop_mine = True if mine_sys.output['drop_mine'] >= 0 else False
        fire = True if fire_sys.output['fire_gun'] >= 0 else False

        # Currently no turning logic implemented
        turn_rate = 0.0

        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        return "Project Controller"
