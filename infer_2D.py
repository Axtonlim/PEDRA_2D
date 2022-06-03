import Bayesian_Hilbert_Maps.BHM.original.sbhm as sbhm
from aux_funcs import *
import agent2D 
import rrt_BHM 
import matplotlib.pyplot as plt

valid_starting_points1 = [(56, 56), (112, 56), (168, 56), (168, 112), (112, 112)]  # X, Y for Drone 1
valid_starting_points2 = [(168, 168), (112, 168), (56, 168), (56, 112), (112, 112)]  # X, Y for Drone 2

# Training map
gt = get_ground_truth_array(r'/Users/axtonlim/Desktop/PEDRA_2D/map_2D/environments/filled_simple_floorplan_v2.png')
plt.imshow(gt, 'Greys_r')
plt.show()

# Paths
custom_load_dir1 = '/Users/axtonlim/Desktop/PEDRA_2D/map_2D/results/weights1/drone_2D_6000'
custom_load_dir2 = '/Users/axtonlim/Desktop/PEDRA_2D/map_2D/results/weights2/drone_2D_6000'
log_dir1 = '/Users/axtonlim/Desktop/PEDRA_2D/map_2D/results/inference1/infer_log.txt'
log_dir2 = '/Users/axtonlim/Desktop/PEDRA_2D/map_2D/results/inference2/infer_log.txt'

# RRT variables
danger_radius = 4
occ_threshold = 0.7

# SBHM variables
gamma = 0.02
cell_res = (12, 12)
min_max = (0, 223, 0, 223)
LIDAR_max_range = 76

BHM = sbhm.SBHM(gamma=gamma, cell_resolution=cell_res, cell_max_min=min_max)

# agent
drone1 = agent2D.agent_2D(BHM=BHM, min_max=min_max, LIDAR_pixel_range=LIDAR_max_range, ground_truth_map=gt, starting_pos=valid_starting_points1[0],
                         plot_dir='', weights_dir='', custom_load=custom_load_dir1)
drone2 = agent2D.agent_2D(BHM=BHM, min_max=min_max, LIDAR_pixel_range=LIDAR_max_range, ground_truth_map=gt, starting_pos=valid_starting_points2[0],
                         plot_dir='', weights_dir='', custom_load=custom_load_dir2)
drone1.collect_data()    # need to do 1 fitting of BHM first before can query
drone2.collect_data()

current_state1 = drone1.get_state()
current_state2 = drone2.get_state()

# Inference Variables
cum_path_length1 = 0
cum_path_length2 = 0
minimum_finished_ratio = 0.78

plt.ion()
plt.show()
plt.scatter(drone1.position[0], drone1.position[1], drone2.position[0], drone2.position[1], cmap='jet')
print("******** INFERENCE BEGINS *********")

while True:
    no_dupe1 = drone1.network_model.action_selection_non_repeat(current_state1, current_state2, drone1.previous_actions)
    print('no dupe action1', no_dupe1[0])

    no_dupe2 = drone2.network_model.action_selection_non_repeat(current_state2, current_state1, drone2.previous_actions)
    print('no dupe action2', no_dupe2[0])

    # RRT* Algo
    startpos1 = drone1.position
    goalpos1 = action_idx_to_coords(no_dupe1[0], min_max)
    startpos2 = drone2.position
    goalpos2 = action_idx_to_coords(no_dupe2[0], min_max)

    G1 = rrt_BHM.Graph(startpos1, goalpos1, min_max)
    G1 = rrt_BHM.RRT_n_star(G1, drone1.BHM, n_iter=450, radius=5, stepSize=14, crash_radius=5, n_retries_allowed=0)

    G2 = rrt_BHM.Graph(startpos2, goalpos2, min_max)
    G2 = rrt_BHM.RRT_n_star(G2, drone2.BHM, n_iter=450, radius=5, stepSize=14, crash_radius=5, n_retries_allowed=0)

    if G1.success:
        path1 = rrt_BHM.dijkstra(G1)

        path1 = [(int(elem[0]), int(elem[1])) for elem in path1]

        _1, path_length1 = drone1.move_by_sequence(path1[1:])  # exclude first point

        cum_path_length1 += path_length1

    else:
        path_length1 = 0

    if G2.success:
        path2 = rrt_BHM.dijkstra(G2)

        path2 = [(int(elem[0]), int(elem[1])) for elem in path2]

        _2, path_length2 = drone2.move_by_sequence(path2[1:])  # exclude first point

        cum_path_length2 += path_length2

    else:
        path_length2 = 0

    done = False
    if path_length1 or path_length2 != 0:
        free_mask1 = drone1.get_free_mask()
        correct1 = np.logical_and(gt, free_mask1)
        free_mask2 = drone2.get_free_mask()
        correct2 = np.logical_and(gt, free_mask2)
        correct = correct1 + correct2
        #plt.imshow(correct, cmap='Greys_r')
        plt.scatter(drone1.position[0], drone1.position[1], drone2.position[0], drone2.position[1], cmap='jet')
        plt.draw()
        plt.pause(0.001)
        #drone1.show_model()
        #drone2.show_model()
        finished_ratio = np.sum(correct) / np.sum(gt)
        print("Finished ratio:", finished_ratio)

        if finished_ratio > minimum_finished_ratio:
            done = True

        new_state1 = drone1.get_state()
        new_state2 = drone2.get_state()

    else:
        new_state1 = current_state1
        new_state2 = current_state2

    if done:
        print("******** EXPLORATION DONE *********")
        cum_path_length = cum_path_length1 + cum_path_length2
        print("Path Length:", cum_path_length)
        print("Finished ratio:", finished_ratio)
        break

    else:
        current_state1 = new_state1
        current_state2 = new_state2
        drone1.previous_actions.add(tuple(no_dupe1[0]))
        drone2.previous_actions.add(tuple(no_dupe2[0]))



