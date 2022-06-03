# FROM THIS POINT OUT, LET COORD BE DESCRIBED AS X, Y.
# To interpret model output, since he gives in terms of 224x224 action space, the first index is actually row (which is Y)
# and the second index is width (which is X)
from ast import Or
import Bayesian_Hilbert_Maps.BHM.original.sbhm as sbhm 
from aux_funcs import *
import agent2D 
import time
import rrt_BHM 
import matplotlib.pyplot as plt

valid_starting_points1 = [(56, 56), (112, 56), (168, 56), (168, 112), (112, 112)]  # X, Y for Drone 1
valid_starting_points2 = [(168, 168), (112, 168), (56, 168), (56, 112), (112, 112)]  # X, Y for Drone 2

# Training map
gt = get_ground_truth_array(r'/Users/axtonlim/Desktop/PEDRA_2D/map_2D/environments/filled_simple_floorplan.png')
#plt.imshow(gt, 'Greys_r')
#plt.show()

# Paths
log_dir = '/Users/axtonlim/Desktop/PEDRA_2D/map_2D/results/log'

plot_dir1 = '/Users/axtonlim/Desktop/PEDRA_2D/map_2D/results/stats1'
weights_dir1 = '/Users/axtonlim/Desktop/PEDRA_2D/map_2D/results/weights1'

plot_dir2 = '/Users/axtonlim/Desktop/PEDRA_2D/map_2D/results/stats2'
weights_dir2 = '/Users/axtonlim/Desktop/PEDRA_2D/map_2D/results/weights2'

custom_load1 = r''
custom_load2 = r''

# Initialise variables
iter1 = 0
iter2 = 0
max_iters = 10000
save_interval = max_iters // 5
level1 = 0   # if implementing switching starting positions
level2 = 0
current_starting_pos_index1 = 0
current_starting_pos_index2 = 0
episode1 = 0  # how many times drone completed exploration
episode2 = 0
moves_taken1 = 0
moves_taken2 = 0
epsilon_saturation = 10000
epsilon_model = 'exponential'
epsilon1 = 0  # start with drone always taking random actions
epsilon2 = 0
cum_return1 = 0
cum_return2 = 0
discount_factor = 0.9
Q_clip = False   # clips TD error to -1, 1
learning_rate = 2e-6

consecutive_fails1 = 0
consecutive_fails2 = 0
max_consecutive_fails = 15  # for debugging purposes

# RRT variables
danger_radius = 5
occ_threshold = 0.7

# SBHM variables
gamma = 0.02
cell_res = (12, 12)
min_max = (0, 223, 0, 223)
LIDAR_max_range = 50    # in pixels

BHM = sbhm.SBHM(gamma=gamma, cell_resolution=cell_res, cell_max_min=min_max)

# agent
drone1 = agent2D.agent_2D(BHM=BHM, min_max=min_max, LIDAR_pixel_range=LIDAR_max_range, ground_truth_map=gt, starting_pos=valid_starting_points1[current_starting_pos_index1],
                         plot_dir=plot_dir1, weights_dir=weights_dir1, custom_load=custom_load1)
drone2 = agent2D.agent_2D(BHM=BHM, min_max=min_max, LIDAR_pixel_range=LIDAR_max_range, ground_truth_map=gt, starting_pos=valid_starting_points2[current_starting_pos_index2],
                         plot_dir=plot_dir2, weights_dir=weights_dir2, custom_load=custom_load2)
drone1.collect_data() 
drone2.collect_data()   # need to do 1 fitting of BHM first before can query
current_state1 = drone1.get_state()
current_state2 = drone2.get_state()

#plt.ion()
#plt.show()
print("******** SIMULATION BEGINS *********")
# TRAINING LOOP
log_file = open(log_dir + '/log.txt', mode='w')

while True:
    start_time = time.time()

    action1, action_type1, epsilon1 = policy_FCQN(epsilon1, current_state1,
                                               iter1, epsilon_saturation, 'exponential', drone1)
    # action, action_type, epsilon = policy_FCQN_no_dupe(epsilon, current_state,
    #                                                    iter, epsilon_saturation, 'exponential', drone)
    action2, action_type2, epsilon2 = policy_FCQN(epsilon2, current_state2,
                                               iter2, epsilon_saturation, 'exponential', drone2)

    drone1.previous_actions.add(tuple(action1[0]))    # TODO: Hide this working into drone class so won't forget to do
    drone2.previous_actions.add(tuple(action2[0]))

    # RRT* algo
    startpos1 = drone1.position
    startpos2 = drone2.position
    goalpos1 = action_idx_to_coords(action1[0], min_max)
    goalpos2 = action_idx_to_coords(action2[0], min_max)

    valid_goal = True

    surroundings1 = bloom(goalpos1, danger_radius, resolution_per_quadrant=16)
    surroundings2 = bloom(goalpos2, danger_radius, resolution_per_quadrant=16)
    pred_occupancies1 = drone1.BHM.predict_proba(surroundings1)[:, 1]
    pred_occupancies2 = drone2.BHM.predict_proba(surroundings2)[:, 1]
    goal_close_to_obstacle1 = any(occ_val1 > occ_threshold for occ_val1 in pred_occupancies1)
    goal_close_to_obstacle2 = any(occ_val2 > occ_threshold for occ_val2 in pred_occupancies2)

    pred_goal1 = drone1.BHM.predict_proba(np.array([goalpos1]))[0][1]
    pred_goal2 = drone2.BHM.predict_proba(np.array([goalpos2]))[0][1]
    goal_in_unknown_space1 = 0.4 < pred_goal1 < 0.65  
    goal_in_unknown_space2 = 0.4 < pred_goal2 < 0.65      # roughly, if my probability of being occupied is around 0.5 +- 0.1, means im unsure, which is dangerous

    if pred_goal1 > occ_threshold or goal_close_to_obstacle1:  # point selected is in obstacle / too close
        path1 = None
        path_length1 = 0
        safe_travel1 = None

    else:
        G1 = rrt_BHM.Graph(startpos1, goalpos1, min_max)
        # G = rrt_BHM.RRT_n_star(G, drone.BHM, n_iter=450, radius=5,      # RRT Params must be modified based on the environment, but this is not an issue of the agent
        #                        stepSize=14, crash_radius=5, n_retries_allowed=0)
        G1 = rrt_BHM.RRT_n_star_np_arr(G1, np.reshape(drone1.BHM.predict_proba(drone1.qX)[:, 1], (224, 224)),
                                      n_iter=500, radius=5, stepSize=14, crash_radius=5, n_retries_allowed=0)
        if G1.success:
            path1 = rrt_BHM.dijkstra(G1)
            path1 = [(int(elem[0]), int(elem[1])) for elem in path1]

            safe_travel1, path_length1 = drone1.move_by_sequence(path1[1:])  # exclude first point

            if path_length1 == 0:
                consecutive_fails1 += 1
                if consecutive_fails1 == max_consecutive_fails:
                    print("DRONE1 STUCKKKK")
                    print('drone1_pos:', drone1.position)
                    print('goal_pos1:', goalpos1)
                    # rrt_BHM.plot(G1, drone1.BHM, None)
                    drone1.reset(fresh_BHM=sbhm.SBHM(gamma=gamma, cell_resolution=cell_res,cell_max_min=min_max),
                                                                    starting_pos1=valid_starting_points1[current_starting_pos_index1])
                    current_state1 = drone1.get_state()
                    # don't +1 to episode, treat as same episode and reset move and return
                    moves_taken1 = 0
                    cum_return1 = 0
                    consecutive_fails1 = 0
                    continue
            else:
                consecutive_fails1 = 0

            moves_taken1 += 1
        else:

            path1 = None
            path_length1 = 0
            safe_travel1 = None

    if pred_goal2 > occ_threshold or goal_close_to_obstacle2:  # point selected is in obstacle / too close
        path2 = None
        path_length2 = 0
        safe_travel2 = None

    else:
        G2 = rrt_BHM.Graph(startpos2, goalpos2, min_max)
        # G = rrt_BHM.RRT_n_star(G, drone.BHM, n_iter=450, radius=5,      # RRT Params must be modified based on the environment, but this is not an issue of the agent
        #                        stepSize=14, crash_radius=5, n_retries_allowed=0)
        G2 = rrt_BHM.RRT_n_star_np_arr(G2, np.reshape(drone2.BHM.predict_proba(drone2.qX)[:, 1], (224, 224)),
                                      n_iter=500, radius=5, stepSize=14, crash_radius=5, n_retries_allowed=0)
        if G2.success:
            path2 = rrt_BHM.dijkstra(G2)
            path2 = [(int(elem[0]), int(elem[1])) for elem in path2]

            safe_travel2, path_length2 = drone2.move_by_sequence(path2[1:])  # exclude first point

            if path_length2 == 0:
                consecutive_fails2 += 1
                if consecutive_fails2 == max_consecutive_fails:
                    print("DRONE2 STUCKKKK")
                    print('drone2_pos:', drone2.position)
                    print('goal_pos2:', goalpos2)
                    # rrt_BHM.plot(G2, drone2.BHM, None)
                    drone2.reset(fresh_BHM=sbhm.SBHM(gamma=gamma, cell_resolution=cell_res,cell_max_min=min_max),
                                                                    starting_pos2=valid_starting_points2[current_starting_pos_index2])
                    current_state2 = drone2.get_state()
                    # don't +1 to episode, treat as same episode and reset move and return
                    moves_taken2 = 0
                    cum_return2 = 0
                    consecutive_fails2 = 0
                    continue
            else:
                consecutive_fails2 = 0

            moves_taken2 += 1
        else:

            path2 = None
            path_length2 = 0
            safe_travel2 = None

    reward1 = drone1.reward_gen(path_length1, path_length2, goal_in_unknown_space=goal_in_unknown_space1, safe_travel=safe_travel1)
    reward2 = drone2.reward_gen(path_length2, path_length1, goal_in_unknown_space=goal_in_unknown_space2, safe_travel=safe_travel2)

    # check for completeness and update state, only if moved
    done1 = False
    done2 = False
    if path_length1 or path_length2 != 0:
        free_mask1 = drone1.get_free_mask()
        free_mask2 = drone2.get_free_mask()
        correct1 = np.logical_and(gt, free_mask1)
        correct2 = np.logical_and(gt, free_mask2)
        correct = correct1 + correct2
        #plt.imshow(correct, cmap='Greys_r')
        #plt.draw()
        #plt.pause(0.001)
        # drone1.show_model()
        # drone2.show_model()
        finished_ratio = np.sum(correct) / np.sum(gt)
        # print("Finished ratio:", finished_ratio)

        if finished_ratio > 0.78:
            done1 = True
            done2 = True
            reward1 += 1
            reward2 += 1

        new_state1 = drone1.get_state()
        new_state2 = drone2.get_state()
    else:
        new_state1 = current_state1
        new_state2 = current_state2

    # TRAINING DONE HERE FOR DRONE 1 & 2
    cum_return1 = cum_return1 + reward1
    cum_return2 = cum_return2 + reward2

    data_tuple1 = (current_state1, action1, new_state1, reward1)
    data_tuple2 = (current_state2, action2, new_state2, reward2)

    _, Q_target, err = get_err_FCQN(data_tuple1, data_tuple2, drone1, discount_factor, Q_clip)
    print("No problem") 
    drone1.network_model.train_n(current_state1, current_state2, action1, Q_target, 1, learning_rate, epsilon1, iter1)
    drone2.network_model.train_n(current_state2, current_state1, action2, Q_target, 1, learning_rate, epsilon2, iter2)
    # ------------------

    time_exec = time.time() - start_time

    s_log1 = 'Drone1 - Level1 {:>2d} - Iter1: {:>5d}/{:<4d} Action1: {}-{:>5s} Eps1: {:<1.4f} Lr1: {:>1.6f} Ret1 = {:<+6.4f} t1={:<1.3f} Moves1: {:<2} Steps1: {:<3} Reward1: {:<+1.4f}  '.format(
        level1,
        iter1,
        episode1,
        action1,
        action_type1,
        epsilon1,
        learning_rate,
        cum_return1,
        time_exec,
        moves_taken1,
        drone1.steps_taken,
        reward1)

    s_log2 = 'Drone2 - Level2 {:>2d} - Iter2: {:>5d}/{:<4d} Action2: {}-{:>5s} Eps2: {:<1.4f} Lr2: {:>1.6f} Ret2 = {:<+6.4f} t2={:<1.3f} Moves2: {:<2} Steps2: {:<3} Reward2: {:<+1.4f}  '.format(
        level2,
        iter2,
        episode2,
        action2,
        action_type2,
        epsilon2,
        learning_rate,
        cum_return2,
        time_exec,
        moves_taken2,
        drone2.steps_taken,
        reward2)

    print(s_log1 + s_log2)
    log_file.write(s_log1 + '\n' + s_log2 + '\n')


    if done1:
        drone1.network_model.log_to_tensorboard(tag='Return', group='Drone1',
                                                           value=cum_return1,
                                                           index=episode1)
        drone1.network_model.log_to_tensorboard(tag='Moves (valid goalpoints)', group='Drone1',
                                                       value=moves_taken1,
                                                       index=episode1)
        drone1.network_model.log_to_tensorboard(tag='Steps (waypoints)', group='Drone1',
                                               value=len(drone1.previous_positions),
                                               index=episode1)

        drone1.reset(fresh_BHM=sbhm.SBHM(gamma=gamma, cell_resolution=cell_res,
                                        cell_max_min=min_max),
                    starting_pos=valid_starting_points1[current_starting_pos_index1])

        current_state1 = drone1.get_state()

        # drone1.show_model()
        episode1 += 1
        moves_taken1 = 0
        cum_return1 = 0

        if episode1 % 3 == 0 and episode1 > 0:    # Change starting points every 3 Episodes
            current_starting_pos_index1 += 1
            if current_starting_pos_index1 == len(valid_starting_points1):
                current_starting_pos_index1 = 0
            level1 = current_starting_pos_index1
            print("Changing starting pos1")

    else:
        current_state1 = new_state1

    if done2:
        drone2.network_model.log_to_tensorboard(tag='Return', group='Drone2',
                                                           value=cum_return2,
                                                           index=episode2)
        drone2.network_model.log_to_tensorboard(tag='Moves (valid goalpoints)', group='Drone2',
                                                       value=moves_taken2,
                                                       index=episode2)
        drone2.network_model.log_to_tensorboard(tag='Steps (waypoints)', group='Drone2',
                                               value=len(drone2.previous_positions),
                                               index=episode2)

        drone2.reset(fresh_BHM=sbhm.SBHM(gamma=gamma, cell_resolution=cell_res,
                                        cell_max_min=min_max),
                    starting_pos=valid_starting_points2[current_starting_pos_index2])

        current_state2 = drone2.get_state()

        # drone2.show_model()
        episode2 += 1
        moves_taken2 = 0
        cum_return2 = 0

        if episode2 % 3 == 0 and episode2 > 0:    # Change starting points every 3 Episodes
            current_starting_pos_index2 += 1
            if current_starting_pos_index2 == len(valid_starting_points2):
                current_starting_pos_index2 = 0
            level2 = current_starting_pos_index2
            print("Changing starting pos2")

    else:
        current_state2 = new_state2

    iter1 += 1
    iter2 += 1

    if iter1 % save_interval == 0 or iter2 % save_interval == 0:
        drone1.network_model.save_network(str(iter1))
        drone2.network_model.save_network(str(iter2))
    if iter1 == max_iters or iter2 == max_iters:
        print("TRAINING DONE")
        break

log_file.close()
