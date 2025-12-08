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
from kesslergame import KesslerController, Scenario, TrainerEnvironment
import EasyGA


# ---------------------------------
# Genetic Algorithm Implementation
# ---------------------------------

def get_sorted_genes(min_val, max_val, num_points):

    if isinstance(min_val, float) or isinstance(max_val, float):
        # Use uniform for floats (like mass_density)
        genes = np.random.uniform(min_val, max_val, num_points)
    else:
        # Use randint for integers (like velocity)
        genes = np.random.randint(min_val, max_val, num_points)
    return np.sort(genes).tolist()

# Chromosome generation function
def generate_full_chromosome():
    chromosome = []
    # Thrust Control - Thrust
    chromosome.extend(get_sorted_genes(-380, 380, 15))

    # Mine Control - Ship Velocity
    chromosome.extend(get_sorted_genes(0, 380, 7))

    # Mine Control - Mass Density
    chromosome.extend(get_sorted_genes(0.0, 0.1, 7))

    # Fire Control - Ammo
    chromosome.extend(get_sorted_genes(0.0, 1.0, 7))

    # Fire Control - Closest Asteroid Distance
    chromosome.extend(get_sorted_genes(0, 1000, 7))

    # Thrust Control - Distance
    chromosome.extend(get_sorted_genes(0, 1000, 9))

    # Turn/Fire Control - Bullet Time
    chromosome.extend(get_sorted_genes(0.0, 1.0, 8))

    return chromosome

# Genetic Fitness Function
# Maximises asteroids hit
def fitness(chromosome):
    try:
        train_scenario = Scenario(name='Train Scenario',
                                num_asteroids=10,
                                ship_states=[
                                    {'position': (
                                        500, 500), 'angle': 90, 'lives': 3, 'team': 1, "mines_remaining": 3},
                                ],
                                map_size=(1000, 800),
                                time_limit=60,
                                ammo_limit_multiplier=0,
                                stop_if_no_ammo=False)
        game_settings = {'perf_tracker': True,
                     'graphics_type': None,
                     'realtime_multiplier': 1,
                     'graphics_obj': None,
                     'frequency': 30}

        game = TrainerEnvironment(settings=game_settings)
        score, _ = game.run(scenario=train_scenario,
                            controllers=[ProjectController(chromosome)])
        total_asteroids_hit = [team.asteroids_hit for team in score.teams]
        return total_asteroids_hit[0]
    except Exception as e:
        print(f"Exception in GA fitness: {e}")
        return 0

# Function to run the genetic algorithm/find best chromosome
def findBestChromosome(population_size=10, generation_goal=3):
    ga = EasyGA.GA()
    ga.chromosome_impl = generate_full_chromosome
    ga.chromosome_length = 60
    ga.population_size = population_size
    ga.target_fitness_type = 'max'
    ga.generation_goal = generation_goal
    ga.fitness_function_impl = fitness
    ga.evolve()
    ga.print_best()
    return ga.population.chromosome_list[0]

# ---------------------------------
# Controller Implementation
# ---------------------------------

class ProjectController(KesslerController):

    def __init__(self, chromosome=None):
        """
        Any variables or initialization desired for the controller can be set up here
        """
        self.eval_frames = 0
        self.max_bullet_count = None

        self.chromosome = [gene.value for gene in chromosome.gene_list]

        self.setup_mine_control()
        self.setup_agro_fire_control()
        self.setup_thrust_avoidance_control()
        self.setup_turn_and_fire_control() 


    # ---------------------- MINE CONTROL ---------------------- #
    def setup_mine_control(self):
        ship_velocity = ctrl.Antecedent(np.arange(0, 381, 1), 'ship_velocity')
        mass_density = ctrl.Antecedent(np.arange(0, 0.1, 0.001), 'mass_density')
        drop_mine = ctrl.Consequent(np.arange(-1, 1, 0.1), 'drop_mine')

        # If chromosome provided, use chromosome values
        if (self.chromosome):
            # 15 - 22 are ship velocity
            sv_genes = self.chromosome[15:22]
            # 22- 29 are mass density
            md_genes = self.chromosome[22:29]

            ship_velocity['S'] = fuzz.zmf(ship_velocity.universe, sv_genes[0], sv_genes[1])
            ship_velocity['M'] = fuzz.trimf(ship_velocity.universe, [sv_genes[2], sv_genes[3], sv_genes[4]])
            ship_velocity['L'] = fuzz.smf(ship_velocity.universe, sv_genes[5], sv_genes[6])

            mass_density['S'] = fuzz.zmf(mass_density.universe, md_genes[0], md_genes[1])
            mass_density['M'] = fuzz.trimf(mass_density.universe, [md_genes[2], md_genes[3], md_genes[4]])
            mass_density['L'] = fuzz.smf(mass_density.universe, md_genes[5], md_genes[6])
        # Use default values
        else:
            ship_velocity['S'] = fuzz.zmf(ship_velocity.universe, 0, 100)
            ship_velocity['M'] = fuzz.trimf(ship_velocity.universe, [60, 180, 280])
            ship_velocity['L'] = fuzz.smf(ship_velocity.universe, 220, 380)

            mass_density['S'] = fuzz.zmf(mass_density.universe, 0, 0.01)
            mass_density['M'] = fuzz.trimf(mass_density.universe, [0, 0.02, 0.04])
            mass_density['L'] = fuzz.smf(mass_density.universe, 0.03, 0.1)

        # Fuzzy set for consequent (drop mine or not)
        drop_mine['N'] = fuzz.trimf(drop_mine.universe, [-1, -1, 0.0])
        drop_mine['Y'] = fuzz.trimf(drop_mine.universe, [0.0, 1, 1])

        # Declare fuzzy rules
        rules = [
            ctrl.Rule(ship_velocity['S'], drop_mine['N']),
            ctrl.Rule(ship_velocity['M'] & (mass_density['S'] | mass_density['M']), drop_mine['N']),
            ctrl.Rule(ship_velocity['L'] & (mass_density['S'] | mass_density['M']), drop_mine['N']),
            ctrl.Rule((ship_velocity['M'] | ship_velocity['L']) & mass_density['L'], drop_mine['Y']),
        ]

        self.mine_control = ctrl.ControlSystem(rules)

    # ---------------------- AGGRESSIVE FIRE CONTROL ---------------------- #
    def setup_agro_fire_control(self):
        ammo = ctrl.Antecedent(np.arange(0, 1, 0.01), 'ammo')
        fire_gun = ctrl.Consequent(np.arange(-1, 1, 0.1), 'fire_gun')
        closest_a_dist = ctrl.Antecedent(np.arange(0, 1000, 1), 'closest_a_dist')

        # If chromosome, use chromosome values
        if (self.chromosome):
            a_genes = self.chromosome[29:36]
            cad_genes = self.chromosome[36:43]

            ammo['L'] = fuzz.zmf(ammo.universe, a_genes[0], a_genes[1])
            ammo['M'] = fuzz.trimf(ammo.universe, [a_genes[2], a_genes[3], a_genes[4]])
            ammo['H'] = fuzz.smf(ammo.universe, a_genes[5], a_genes[6])

            closest_a_dist['S'] = fuzz.zmf(closest_a_dist.universe, cad_genes[0], cad_genes[1])
            closest_a_dist['M'] = fuzz.trimf(closest_a_dist.universe, [cad_genes[2], cad_genes[3], cad_genes[4]])
            closest_a_dist['L'] = fuzz.smf(closest_a_dist.universe, cad_genes[5], cad_genes[6])
        # Use default values
        else:
            ammo['L'] = fuzz.zmf(ammo.universe, 0, 0.2)
            ammo['M'] = fuzz.trimf(ammo.universe, [0, 0.4, 0.7])
            ammo['H'] = fuzz.smf(ammo.universe, 0.4, 1.0)

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

        self.agro_fire_control = ctrl.ControlSystem(rules)

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
        distance = ctrl.Antecedent(np.arange(0, 1001, 1), 'distance')
        vert_offset = ctrl.Antecedent(np.arange(-500, 501, 1), 'vert_offset')

        # Fuzzy output variable
        thrust = ctrl.Consequent(np.arange(-380, 381, 1), 'thrust')

        # Vertical offset membership functions
        vert_offset['below'] = fuzz.trimf(vert_offset.universe, [-300, -300, 0])
        vert_offset['center'] = fuzz.trimf(vert_offset.universe, [-50, 0, 50])
        vert_offset['above'] = fuzz.trimf(vert_offset.universe, [0, 300, 300])

         # If chromosome, use chromosome values
        if (self.chromosome):
            t_genes = self.chromosome[0:15]
            d_genes = self.chromosome[43:52]

            distance['near'] = fuzz.trimf(distance.universe, [d_genes[0], d_genes[1], d_genes[2]])
            distance['mid'] = fuzz.trimf(distance.universe, [d_genes[3], d_genes[4], d_genes[5]])
            distance['far'] = fuzz.trimf(distance.universe, [d_genes[6], d_genes[7], d_genes[8]])

            thrust['high_down']   = fuzz.trimf(thrust.universe, [t_genes[0], t_genes[1], t_genes[2]])
            thrust['medium_down'] = fuzz.trimf(thrust.universe, [t_genes[3], t_genes[4], t_genes[5]])
            thrust['none']        = fuzz.trimf(thrust.universe, [t_genes[6], t_genes[7], t_genes[8]])
            thrust['medium_up']   = fuzz.trimf(thrust.universe, [t_genes[9],  t_genes[10], t_genes[11]])
            thrust['high_up']     = fuzz.trimf(thrust.universe, [t_genes[12], t_genes[13], t_genes[14]])
        # Use default values
        else:
            distance['near'] = fuzz.trimf(distance.universe, [0, 0, 300])
            distance['mid'] = fuzz.trimf(distance.universe, [100, 500, 800])
            distance['far'] = fuzz.trimf(distance.universe, [500, 1000, 1000])

            thrust['high_down']   = fuzz.trimf(thrust.universe, [-380, -380, -140])
            thrust['medium_down'] = fuzz.trimf(thrust.universe, [-280, -160, -40])
            thrust['none']        = fuzz.trimf(thrust.universe, [-40, 0,   40])
            thrust['medium_up']   = fuzz.trimf(thrust.universe, [40,  160, 280])
            thrust['high_up']     = fuzz.trimf(thrust.universe, [140, 380, 380])

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

    # Note: The first three methods are directly from Dr. Scott's Implementation for targeting closest Asteroid
    def setup_turn_and_fire_control(self):
        # Declare variables
        bullet_time = ctrl.Antecedent(np.arange(0,1.0,0.002), 'bullet_time')
        theta_delta = ctrl.Antecedent(np.arange(-1*math.pi/30,math.pi/30,0.1), 'theta_delta') # Radians due to Python
        ship_turn = ctrl.Consequent(np.arange(-180,180,1), 'ship_turn') # Degrees due to Kessler
        ship_fire = ctrl.Consequent(np.arange(-1,1,0.1), 'ship_fire')
        
        # If chromosome provided, use chromosome values
        if (self.chromosome):
            bt_genes = self.chromosome[52:60]

            bullet_time['S'] = fuzz.trimf(bullet_time.universe,[bt_genes[0], bt_genes[1], bt_genes[2]])
            bullet_time['M'] = fuzz.trimf(bullet_time.universe, [bt_genes[3], bt_genes[4], bt_genes[5]])
            bullet_time['L'] = fuzz.smf(bullet_time.universe, bt_genes[6], bt_genes[7])
        # Use default values
        else:
            bullet_time['S'] = fuzz.trimf(bullet_time.universe,[0, 0, 0.05])
            bullet_time['M'] = fuzz.trimf(bullet_time.universe, [0, 0.05, 0.1])
            bullet_time['L'] = fuzz.smf(bullet_time.universe, 0.0, 0.1)
        
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
        - tca: time to closest approach (project position onto relative velocity)
        - distance_at_tca: distance between ship and asteroid at tca
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

            # CHANGES MADE

            # TCA (Time of Closest Approach) - Time at which Asteroid is Closest to ship ; Initialized to 0
            tca = 0
            # Cannot calculate tca if relative velocity is zero (denominator = 0)
            denominator = dvx**2 + dvy**2

            # Initial Values for storing threat info of each asteroid
            threat_info = {
                "asteroid": asteroid,
                "time_to_collision": None,
                "distance_at_tca": None,
                "current_distance": math.sqrt(dx**2 + dy**2),
                "threat_level": 0,
            }

            # Define safe distance based on asteroid size - larger asteroids can break into smaller ones
            if asteroid['radius'] >= 16:
                safe_distance = asteroid["radius"] + 50
            else:
                safe_distance = asteroid["radius"] + 30

            # If relative velocity is extremely small, asteroid is approaching very slowly - handle based on distance alone
            if denominator < 1e-1:
                tca = 0         # closest approach is effectively now
                distance_at_tca = math.sqrt(dx*dx + dy*dy)
                
                # Dangerous if within safety boundary
                if distance_at_tca < safe_distance:
                    threat_info["time_to_collision"] = 0
                    threat_info["distance_at_tca"]  = distance_at_tca
                    # threat_info["current_distance"] = distance_at_tca
                    threat_info["threat_level"] = 2
                    critical_threats.append(threat_info)
                    continue
                else:
                    safe_targets.append(threat_info)
                    continue

            if denominator > 0:
                # r = (dx,dy) is the vector from ship to asteroid and v_rel = (dvx,dvy) is relative velocity.
                # This comes from projecting r onto v_rel: it finds the time when the separation vector is perpendicular to the relative velocity (i.e., closest point).
                # max(0, ...) prevents negative times. If the projection gives a negative result, the closest-approach time is in the past, so we treat the closest approach as right now (t = 0).
                # If the asteroid is heading toward your ship, the dot product r·v_rel will be negative and tca positive. If it's moving away, tca will be negative.
                tca = max(0, -(dx * dvx + dy * dvy) / denominator)
                threat_info["time_to_collision"] = tca

            # Distance between ship and asteroid at that closest approach time
            distance_at_tca = math.sqrt((dx + dvx * tca) ** 2 + (dy + dvy * tca) ** 2)
            threat_info["distance_at_tca"] = distance_at_tca

            # If they come closer than our safe distance, classify by how soon that occurs.
            if distance_at_tca < safe_distance:
                if tca < 1:  # Critical: collision in less than 2 seconds
                    threat_info["threat_level"] = 2
                    critical_threats.append(threat_info)
                elif tca < 2:  # Moderate: collision in 2-3 seconds
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
        # Calculate evasion maneuver when threats are present ; Returns: (thrust, turn_rate, fire_while_evading)
        
        # No threats, no evasion needed
        if not critical_threats and not moderate_threats:
            return 0, 0, True  # No evasion needed
        
        # Ship state
        ship_x, ship_y = ship_state["position"]
        ship_heading = math.radians(ship_state["heading"])
        
        # Combine all threats, Each threat is weighted - critical threats are three times more important than moderate ones.
        all_threats = []
        for threat in critical_threats:
            all_threats.append((threat, 3.0))  # High weight for critical
        for threat in moderate_threats:
            all_threats.append((threat, 1.0))  # Lower weight for moderate
        
        # Calculate threat vector (weighted average of threat directions)
        threat_vector_x = 0
        threat_vector_y = 0
        total_weight = 0
        
        # Each threat represents a single asteroid
        for threat, weight in all_threats:
            asteroid = threat["asteroid"]
            dx = asteroid["position"][0] - ship_x
            dy = asteroid["position"][1] - ship_y
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance > 0:
                # Proximity factor to prioritize closer threats - pf larger when asteroid is closer
                proximity_factor = 1.0 / (distance + 1) 
                # Normalize direction, scale by weight (3 for critical, 1 for moderate) and proximity (closeness)
                # Basically, Add a contribution to the overall threat vector in the direction of this asteroid, scaled by how important it is and how close it is.
                threat_vector_x += (dx / distance) * weight * proximity_factor  # threat contrib. of this asteroid in x direction
                threat_vector_y += (dy / distance) * weight * proximity_factor  # threat contrib. of this asteroid in y direction
                # Keeps track of sum of all threat contributions for normalization of threat vector later (to get an average direction)
                total_weight += weight * proximity_factor  # total threat weight contribution from this asteroid
        
        # If we have any threats, compute evasion direction
        if total_weight > 0:
            # Normalize threat vector to get average threat direction
            threat_vector_x /= total_weight # average x component of threat direction
            threat_vector_y /= total_weight # average y component of threat direction
            # Calculate angle of threat vector - basically, an angle pointing toward the combined threat direction - 'center of all threats'
            threat_direction = math.atan2(threat_vector_y, threat_vector_x)
            
            # Calculate evasion direction (perpendicular to threat) - rotates the threat vector by 90° clockwise
            evasion_direction = threat_direction + math.pi/2
            
            # Calculate angle difference between current heading and evasion direction
            current_to_evasion = evasion_direction - ship_heading
            # Wrap to (-pi, pi) - to get the smallest angle difference ie. shortest turn direction required
            current_to_evasion = (current_to_evasion + math.pi) % (2 * math.pi) - math.pi

            # Choose the evasion direction that requires less turning 
            # If turning left is easier, use left evasion instead
            if abs(current_to_evasion) > math.pi/2:
                evasion_direction = threat_direction - math.pi/2  # 90 degrees left
                current_to_evasion = evasion_direction - ship_heading
                current_to_evasion = (current_to_evasion + math.pi) % (2 * math.pi) - math.pi
            
            # Calculate turn rate
            # Converts angle difference to degrees, multiplies by 30 to scale it to turn rate, and clamps to +-180°
            turn_rate = np.clip(math.degrees(current_to_evasion) * 30, -180, 180)
            
            # Calculate thrust - more aggressive for critical threats
            base_thrust = 200 if critical_threats else 75
            # Scale thrust based on number of critical threats (more threats - more thrust)
            thrust = base_thrust * (1.0 + min(1.0, len(critical_threats) * 0.5))
            if thrust > 300:
                thrust = 380  # Cap thrust to max
            
            # Only fire while evading if we have moderate threats but no critical ones
            fire_while_evading = (len(critical_threats) == 0 and len(moderate_threats) > 0)
            
            return thrust, turn_rate, fire_while_evading
        
        return 0, 0, True

    def calc_dist(self, ship_pos_x, ship_pos_y, asteroid):
        curr_dist = math.sqrt((ship_pos_x - asteroid["position"][0])**2 + (ship_pos_y - asteroid["position"][1])**2)
        return curr_dist

# MAIN CONTROLLER ACTION METHOD

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        
        # Collect required initial data
        ship_x = ship_state["position"][0]
        ship_y = ship_state["position"][1]
        ship_vel_x, ship_vel_y = ship_state["velocity"]
        current_ammo = ship_state["bullets_remaining"]

        asteroid_data = self.collect_asteroid_data(game_state["asteroids"], ship_x, ship_y)
        mass_density = asteroid_data["mass_density"]
        closest_asteroid = asteroid_data["closest_asteroid"]
        ship_vel_magnitude = math.sqrt((ship_vel_x) ** 2 + (ship_vel_y ** 2))

        if self.max_bullet_count is None:
            # Avoid division by zero later
            self.max_bullet_count = current_ammo if current_ammo > 0 else 1

        # Ammo ratio: if infinite ammo (-1), treat as full (1.0)
        ammo_ratio = 0.0
        if current_ammo == -1:
            ammo_ratio = 1.0
        else:
            ammo_ratio = current_ammo / self.max_bullet_count if self.max_bullet_count > 0 else 0.0

        # Categorize all asteroids by threat level
        critical_threats, moderate_threats, safe_targets = self.categorize_asteroids(ship_state, game_state)
        
        # PRIORITY 1: Evasion if threats exist
        if critical_threats or moderate_threats:
            thrust, turn_rate, fire_while_evading = self.calculate_evasion_maneuver(ship_state, critical_threats, moderate_threats)
            
            # If we can fire while evading, target the most dangerous shootable asteroid
            if fire_while_evading and safe_targets:
                best_target_old = safe_targets[0]  # Closest safe target
                best_target = {"aster": None, "dist": float('inf')}
                best_target["aster"] = safe_targets[0]["asteroid"]
                best_target["dist"]  = self.calc_dist(ship_x, ship_y, best_target["aster"])

                targeting_turn_rate, targeting_fire = self.find_turn_rate_fire(ship_x, ship_y, best_target, ship_state)
                # Blend evasion and targeting (leaning toward evasion)
                turn_rate = 0.7 * turn_rate + 0.3 * targeting_turn_rate
                fire = targeting_fire
            else:
                fire = True
        
        # PRIORITY 2: Offensive targeting when safe
        else:
            # Use avoidance thrust system to stay safe & target closest asteroid
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
                turn_rate = self.find_turn_rate_fire(ship_x, ship_y, closest_asteroid, ship_state)[0]

                # Use aggressive fire system when ship is safe
                agro_fire = ctrl.ControlSystemSimulation(self.agro_fire_control, flush_after_run=1)
                agro_fire.inputs({
                    'ammo': ammo_ratio,
                    'closest_a_dist': closest_asteroid["dist"]
                })
                agro_fire.compute()
                fire = True if agro_fire.output['fire_gun'] >= 0 else False
            else:
                turn_rate, fire = 0, False

            if abs(turn_rate) > 120:
                fire = False  # stop shooting when making big turns

        # Mine drop logic
        mine_sys = ctrl.ControlSystemSimulation(self.mine_control, flush_after_run=1)
        mine_sys.inputs({
            'mass_density': mass_density,
            'ship_velocity': ship_vel_magnitude
        })
        mine_sys.compute()
        drop_mine = True if mine_sys.output['drop_mine'] >= 0 else False

        self.eval_frames += 1
        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        return "Project Controller"
    

