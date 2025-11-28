# ECE449 Fall 2025
# Project Controller
#
# Jeremy Carefoot
# Hamdan Raashid
# Zeeshan Haque

import math
import numpy as np
import skfuzzy as fuzz
from typing import Dict, Tuple
from skfuzzy import control as ctrl
from kesslergame import KesslerController


class ProjectController(KesslerController):

    def __init__(self):
        """
        Any variables or initialization desired for the controller can be set up here
        """
        self.eval_frames = 0
        self.max_bullet_count = None

        self.setup_mine_control()
        self.setup_fire_control()
        self.setup_thrust_avoidance_control()
        self.setup_turn_and_fire_control() 

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
            ay = a["position"][1]
            ax = a["position"][0]
            # euclidian distance
            curr_dist = math.hypot(ship_x - ax, ship_y - ay)

            if closest_asteroid is None:
                closest_asteroid = dict(aster = a, dist = curr_dist, vert_offset = ay - ship_y)

            if curr_dist < DENSITY_RADIUS:
                total_mass += a["mass"]
                
            else:    
                # closest_asteroid exists, and is thus initialized. 
                if closest_asteroid["dist"] > curr_dist:
                    # New minimum found
                    closest_asteroid["aster"] = a
                    closest_asteroid["dist"] = curr_dist
                    closest_asteroid["vert_offset"] = ay - ship_y

        density_area = math.pi * (DENSITY_RADIUS ** 2)
        density = total_mass / density_area if density_area > 0 else 0.0

        return {
            "closest_asteroid": closest_asteroid,
            "mass_density": density
        }

    def setup_thrust_avoidance_control(self):
        # Fuzzy input variables
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

        # Define fuzzy rules
        rules = [
            ctrl.Rule(distance['near'] & vert_offset['above'], thrust['high_down']),
            ctrl.Rule(distance['near'] & vert_offset['below'], thrust['high_up']),
            ctrl.Rule(distance['mid'] & vert_offset['above'], thrust['medium_down']),
            ctrl.Rule(distance['mid'] & vert_offset['below'], thrust['medium_up']),
            ctrl.Rule(distance['far'], thrust['none']),
            ctrl.Rule(vert_offset['center'], thrust['none'])
        ]

        # Create the fuzzy control system and simulation
        self.avoidance_control = ctrl.ControlSystem(rules)
        self.avoidance_sim = ctrl.ControlSystemSimulation(self.avoidance_control)

# HR IMPLEMENTATION

    # Note: The first three methods are directly from Dr. Scott's Implementation for targeting closest Asteroid

    def setup_turn_and_fire_control(self):
        # self.targeting_control -- fuzzy control system for turning and firing - already inherited

        # Declare variables
        bullet_time = ctrl.Antecedent(np.arange(0,1.0,0.002), 'bullet_time')
        theta_delta = ctrl.Antecedent(np.arange(-1*math.pi/30,math.pi/30,0.1), 'theta_delta') # Radians due to Python
        ship_turn = ctrl.Consequent(np.arange(-180,180,1), 'ship_turn') # Degrees due to Kessler
        ship_fire = ctrl.Consequent(np.arange(-1,1,0.1), 'ship_fire')
        
        # Declare fuzzy sets for bullet_time (how long it takes for the bullet to reach the intercept point)
        bullet_time['S'] = fuzz.trimf(bullet_time.universe,[0,0,0.05])
        bullet_time['M'] = fuzz.trimf(bullet_time.universe, [0,0.05,0.1])
        bullet_time['L'] = fuzz.smf(bullet_time.universe,0.0,0.1)
        
        # Declare fuzzy sets for theta_delta (degrees of turn needed to reach the calculated firing angle)
        # Hard-coded for a game step of 1/30 seconds
        theta_delta['NL'] = fuzz.zmf(theta_delta.universe, -1*math.pi/30,-2*math.pi/90)
        theta_delta['NM'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/30, -2*math.pi/90, -1*math.pi/90])
        theta_delta['NS'] = fuzz.trimf(theta_delta.universe, [-2*math.pi/90,-1*math.pi/90,math.pi/90])
        # theta_delta['Z'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/90,0,math.pi/90])
        theta_delta['PS'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/90,math.pi/90,2*math.pi/90])
        theta_delta['PM'] = fuzz.trimf(theta_delta.universe, [math.pi/90,2*math.pi/90, math.pi/30])
        theta_delta['PL'] = fuzz.smf(theta_delta.universe,2*math.pi/90,math.pi/30)
        
        # Declare fuzzy sets for the ship_turn consequent; this will be returned as turn_rate.
        # Hard-coded for a game step of 1/30 seconds
        ship_turn['NL'] = fuzz.trimf(ship_turn.universe, [-180,-180,-120])
        ship_turn['NM'] = fuzz.trimf(ship_turn.universe, [-180,-120,-60])
        ship_turn['NS'] = fuzz.trimf(ship_turn.universe, [-120,-60,60])
        # ship_turn['Z'] = fuzz.trimf(ship_turn.universe, [-60,0,60])
        ship_turn['PS'] = fuzz.trimf(ship_turn.universe, [-60,60,120])
        ship_turn['PM'] = fuzz.trimf(ship_turn.universe, [60,120,180])
        ship_turn['PL'] = fuzz.trimf(ship_turn.universe, [120,180,180])
        
        # Declare singleton fuzzy sets for the ship_fire consequent; -1 -> don't fire, +1 -> fire; this will be  thresholded
        # and returned as the boolean 'fire'
        ship_fire['N'] = fuzz.trimf(ship_fire.universe, [-1,-1,0.0])
        ship_fire['Y'] = fuzz.trimf(ship_fire.universe, [0.0,1,1]) 
                
        # Declare each fuzzy rule
        rules = [
            ctrl.Rule(bullet_time['L'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N'])),
            ctrl.Rule(bullet_time['L'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['N'])),
            ctrl.Rule(bullet_time['L'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y'])),

            ctrl.Rule(bullet_time['L'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['L'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['N'])),
            ctrl.Rule(bullet_time['L'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N'])),
            ctrl.Rule(bullet_time['M'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N'])),
            ctrl.Rule(bullet_time['M'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['N'])),
            ctrl.Rule(bullet_time['M'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y'])),

            ctrl.Rule(bullet_time['M'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['M'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['N'])),
            ctrl.Rule(bullet_time['M'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y'])),

            ctrl.Rule(bullet_time['S'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['Y']))
        ]

        self.targeting_control = ctrl.ControlSystem(rules)
    
    def find_turn_rate_fire(self, ship_pos_x, ship_pos_y, closest_asteroid, ship_state: Dict):
        asteroid_ship_x = ship_pos_x - closest_asteroid["aster"]["position"][0] # type: ignore
        asteroid_ship_y = ship_pos_y - closest_asteroid["aster"]["position"][1] # type: ignore
        asteroid_ship_theta = math.atan2(asteroid_ship_y,asteroid_ship_x)

        asteroid_direction = math.atan2(closest_asteroid["aster"]["velocity"][1], closest_asteroid["aster"]["velocity"][0]) # Velocity is a 2-element array [vx,vy].
        my_theta2 = asteroid_ship_theta - asteroid_direction
        cos_my_theta2 = math.cos(my_theta2)
        # Need the speeds of the asteroid and bullet. speed * time is distance to the intercept point
        asteroid_vel = math.sqrt(closest_asteroid["aster"]["velocity"][0]**2 + closest_asteroid["aster"]["velocity"][1]**2)
        bullet_speed = 800 # Hard-coded bullet speed from bullet.py
        
        # Determinant of the quadratic formula b^2-4ac
        targ_det = (-2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2)**2 - (4*(asteroid_vel**2 - bullet_speed**2) * (closest_asteroid["dist"]**2))
        
        # Combine the Law of Cosines with the quadratic formula for solve for intercept time. Remember, there are two values produced.
        intrcpt1 = ((2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2) + math.sqrt(targ_det)) / (2 * (asteroid_vel**2 -bullet_speed**2))
        intrcpt2 = ((2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2) - math.sqrt(targ_det)) / (2 * (asteroid_vel**2-bullet_speed**2))
        
        # Take the smaller intercept time, as long as it is positive; if not, take the larger one.
        if intrcpt1 > intrcpt2:
            if intrcpt2 >= 0:
                bullet_t = intrcpt2
            else:
                bullet_t = intrcpt1
        else:
            if intrcpt1 >= 0:
                bullet_t = intrcpt1
            else:
                bullet_t = intrcpt2
                
        intrcpt_x = closest_asteroid["aster"]["position"][0] + closest_asteroid["aster"]["velocity"][0] * (bullet_t+1/30)
        intrcpt_y = closest_asteroid["aster"]["position"][1] + closest_asteroid["aster"]["velocity"][1] * (bullet_t+1/30)

        
        my_theta1 = math.atan2((intrcpt_y - ship_pos_y),(intrcpt_x - ship_pos_x))
        
        # Lastly, find the difference betwwen firing angle and the ship's current orientation. BUT THE SHIP HEADING IS IN DEGREES.
        shooting_theta = my_theta1 - ((math.pi/180)*ship_state["heading"])
        
        # Wrap all angles to (-pi, pi)
        shooting_theta = (shooting_theta + math.pi) % (2 * math.pi) - math.pi
        
        # Pass the inputs to the rulebase and fire it
        shooting = ctrl.ControlSystemSimulation(self.targeting_control,flush_after_run=1)
        
        shooting.input['bullet_time'] = bullet_t
        shooting.input['theta_delta'] = shooting_theta
        
        shooting.compute()

        # Get the defuzzified outputs
        turn_rate = shooting.output['ship_turn']
        
        if shooting.output['ship_fire'] >= 0:
            fire = True
        else:
            fire = False
               
        # And return your three outputs to the game simulation. Controller algorithm complete.
        return turn_rate, fire

    # These functions may need to be adjusted a bit more

    def categorize_asteroids(self, ship_state: Dict, game_state: Dict) -> Tuple[list, list, list]:
        """
        Analyze each asteroid and split them into three groups:

        - critical_threats: asteroids that are on a collision path very soon
        - moderate_threats: asteroids that might collide in the near future
        - safe_targets: asteroids not currently on a collision course

        The function computes simple kinematic predictions:
        - dx, dy: vector from ship to asteroid
        - dvx, dvy: relative velocity (asteroid - ship)
        - tca: time to closest approach (project position onto relative velocity) -- Not sure
        - distance_at_tca: distance between ship and asteroid at tca

        We use a safety buffer (asteroid radius + 30) to be conservative when
        deciding whether an asteroid is dangerous.
        """
        ship_x, ship_y = ship_state["position"]
        ship_vel_x, ship_vel_y = ship_state["velocity"]

        critical_threats = []
        moderate_threats = []
        safe_targets = []

        for asteroid in game_state["asteroids"]:
            # Vector from ship to asteroid (position difference)
            dx = asteroid["position"][0] - ship_x
            dy = asteroid["position"][1] - ship_y

            # Relative velocity (asteroid relative to ship)
            dvx = asteroid["velocity"][0] - ship_vel_x
            dvy = asteroid["velocity"][1] - ship_vel_y

            # Compute time to closest approach (tca). If relative velocity is
            # zero (denominator==0) we cannot compute a meaningful tca, so
            # we leave it at 0 which conservatively treats current geometry.
            tca = 0
            denominator = dvx**2 + dvy**2
            if denominator > 0:
                # Projection formula for when distance is smallest; clamp to
                # zero so we don't consider times in the past.
                tca = max(0, -(dx * dvx + dy * dvy) / denominator)

            # Distance between ship and asteroid at that closest approach time
            distance_at_tca = math.sqrt((dx + dvx * tca) ** 2 + (dy + dvy * tca) ** 2)

            # A small safety buffer around the asteroid to be extra cautious
            safe_distance = asteroid["radius"] + 25

            # Package values so callers can inspect the reason for classification
            threat_info = {
                "asteroid": asteroid,
                "time_to_collision": tca,
                "distance_at_tca": distance_at_tca,
                "current_distance": math.sqrt(dx**2 + dy**2),
                "threat_level": 0,
            }

            # If they come closer than our safe distance, classify by how
            # soon that occurs.
            if distance_at_tca < safe_distance:
                if tca < 1.0:  # Critical: collision in less than 1 second
                    threat_info["threat_level"] = 2
                    critical_threats.append(threat_info)
                elif tca < 3.0:  # Moderate: collision in 1-3 seconds
                    threat_info["threat_level"] = 1
                    moderate_threats.append(threat_info)
            else:
                # Safe target - no collision course
                safe_targets.append(threat_info)
        
        # Sort by most dangerous first
        critical_threats.sort(key=lambda x: x["time_to_collision"])
        moderate_threats.sort(key=lambda x: x["time_to_collision"])
        safe_targets.sort(key=lambda x: x["current_distance"])
        
        return critical_threats, moderate_threats, safe_targets
    
    def calculate_evasion_maneuver(self, ship_state: Dict, critical_threats: list, moderate_threats: list) -> Tuple[float, float, bool]:
        """
        Calculate evasion maneuver when threats are present
        Returns: (thrust, turn_rate, fire_while_evading)
        """
        if not critical_threats and not moderate_threats:
            return 0, 0, True  # No evasion needed
        
        ship_x, ship_y = ship_state["position"]
        ship_heading = math.radians(ship_state["heading"])
        
        # Combine all threats, weighted by criticality
        all_threats = []
        for threat in critical_threats:
            all_threats.append((threat, 3.0))  # High weight for critical
        for threat in moderate_threats:
            all_threats.append((threat, 1.0))  # Lower weight for moderate
        
        # Calculate threat vector (weighted average of threat directions)
        threat_vector_x, threat_vector_y = 0, 0
        total_weight = 0
        
        for threat, weight in all_threats:
            asteroid = threat["asteroid"]
            dx = asteroid["position"][0] - ship_x
            dy = asteroid["position"][1] - ship_y
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance > 0:
                # Normalize and weight by threat level and proximity
                proximity_factor = 1.0 / (distance + 1)
                threat_vector_x += (dx / distance) * weight * proximity_factor
                threat_vector_y += (dy / distance) * weight * proximity_factor
                total_weight += weight * proximity_factor
        
        if total_weight > 0:
            threat_vector_x /= total_weight
            threat_vector_y /= total_weight
            threat_direction = math.atan2(threat_vector_y, threat_vector_x)
            
            # Calculate evasion direction (perpendicular to threat)
            evasion_direction = threat_direction + math.pi/2  # 90 degrees right
            
            # Choose the evasion direction that requires less turning
            current_to_evasion = evasion_direction - ship_heading
            current_to_evasion = (current_to_evasion + math.pi) % (2 * math.pi) - math.pi
            
            # If turning left is easier, use left evasion instead
            if abs(current_to_evasion) > math.pi/2:
                evasion_direction = threat_direction - math.pi/2  # 90 degrees left
                current_to_evasion = evasion_direction - ship_heading
                current_to_evasion = (current_to_evasion + math.pi) % (2 * math.pi) - math.pi
            
            # Calculate turn rate
            turn_rate = np.clip(math.degrees(current_to_evasion) * 30, -180, 180)
            
            # Calculate thrust - more aggressive for critical threats
            base_thrust = 200 if critical_threats else 100
            thrust = base_thrust * (1.0 + min(1.0, len(critical_threats) * 0.5))
            
            # Only fire while evading if we have moderate threats but no critical ones
            fire_while_evading = (len(critical_threats) == 0 and len(moderate_threats) > 0)
            
            return thrust, turn_rate, fire_while_evading
        
        return 0, 0, True

    def calc_dist(self, ship_pos_x, ship_pos_y, asteroid):
        curr_dist = math.sqrt((ship_pos_x - asteroid["position"][0])**2 + (ship_pos_y - asteroid["position"][1])**2)
        return curr_dist

# MAIN CONTROLLER ACTION METHOD

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        
        ship_x = ship_state["position"][0]
        ship_y = ship_state["position"][1]
        current_ammo = ship_state["bullets_remaining"]

        if self.max_bullet_count is None:
            # Avoid division by zero later
            self.max_bullet_count = current_ammo if current_ammo > 0 else 1

        asteroid_data = self.collect_asteroid_data(game_state["asteroids"], ship_x, ship_y)
        mass_density = asteroid_data["mass_density"]
        closest_asteroid = asteroid_data["closest_asteroid"]

        # Categorize all asteroids by threat level
        critical_threats, moderate_threats, safe_targets = self.categorize_asteroids(ship_state, game_state)
        
        # PRIORITY 1: Evasion if threats exist
        if critical_threats or moderate_threats:
            thrust, turn_rate, fire_while_evading = self.calculate_evasion_maneuver(
                ship_state, critical_threats, moderate_threats
            )
            
            # If we can fire while evading, target the most dangerous shootable asteroid
            if fire_while_evading and safe_targets:
                best_target_old = safe_targets[0]  # Closest safe target
                best_target = {"aster": None, "dist": float('inf')}
        
                best_target["aster"] = safe_targets[0]["asteroid"]
                best_target["dist"]  = self.calc_dist(ship_x, ship_y, best_target["aster"])

                targeting_turn_rate, targeting_fire = self.find_turn_rate_fire(
                    ship_x, ship_y, best_target, ship_state
                )
                # Blend evasion and targeting (leaning toward evasion)
                turn_rate = 0.7 * turn_rate + 0.3 * targeting_turn_rate
                fire = targeting_fire
            else:
                fire = True
        
        # PRIORITY 2: Offensive targeting when safe
        else:
            # Use existing avoidance for thrust (but less aggressive when safe)
            nearest_dist = closest_asteroid["dist"]
            vert_offset = closest_asteroid["vert_offset"]
            if nearest_dist is None:
                thrust = 0.0
            else:
                self.avoidance_sim.input['distance'] = nearest_dist
                self.avoidance_sim.input['vert_offset'] = vert_offset
                self.avoidance_sim.compute()
                thrust = self.avoidance_sim.output['thrust'] * 0.2  # Less aggressive when safe
            
            # Target the closest asteroid
            if closest_asteroid is not None:
                turn_rate, fire = self.find_turn_rate_fire(ship_x, ship_y, closest_asteroid, ship_state)
            else:
                turn_rate, fire = 0, False
        closest_asteroid = asteroid_data["closest_asteroid"]
        mass_density = asteroid_data["mass_density"]

        # Ammo ratio: if infinite ammo (-1), treat as full (1.0)
        if current_ammo == -1:
            ammo_ratio = 1.0
        else:
            ammo_ratio = current_ammo / self.max_bullet_count if self.max_bullet_count > 0 else 0.0

        mine_sys = ctrl.ControlSystemSimulation(self.mine_control, flush_after_run=1)
        fire_sys = ctrl.ControlSystemSimulation(self.fire_control, flush_after_run=1)

        # Mine deployment logic
        # ammo_ratio = current_ammo / self.max_bullet_count if current_ammo != -1 else 1.0

        mine_sys = ctrl.ControlSystemSimulation(self.mine_control, flush_after_run=1)
        mine_sys.inputs({
            'mass_density': mass_density,
            'ship_thrust': abs(thrust)
        })
        mine_sys.compute()
        drop_mine = True if mine_sys.output['drop_mine'] >= 0 else False

        self.eval_frames += 1
        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        return "Project Controller"
