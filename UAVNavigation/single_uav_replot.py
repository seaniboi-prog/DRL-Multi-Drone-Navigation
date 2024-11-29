from utils import *

algorithm = "ppo"
waypoint_variant = "obstacle"
env_var = "airsim"


shortest_routes = dict()

cont_file = f"routes/{algorithm}/{waypoint_variant}/cont_{env_var}_shortest_route.pkl"
shortest_routes["cont"] = load_obj_file(cont_file)

disc_file = f"routes/{algorithm}/{waypoint_variant}/disc_{env_var}_shortest_route.pkl"
shortest_routes["disc"] = load_obj_file(disc_file)

# Must start with origin
waypoint_variants = dict()

waypoint_variants["single"] = [
    np.array([0.0, 0.0, 0.0], dtype=np.float32),
    np.array([-10.0, -100.0, 5.0], dtype=np.float32)
]

waypoint_variants["multiple"] = [
    np.array([0.0, 0.0, 0.0], dtype=np.float32),
    np.array([15.0, 30.0, 5.0], dtype=np.float32),
    np.array([70.0, 35.0, 5.0], dtype=np.float32),
    np.array([75.0, 0.0, 5.0], dtype=np.float32),
    np.array([70.0, -35.0, 5.0], dtype=np.float32),
    np.array([15.0, -35.0, 5.0], dtype=np.float32),
]

waypoint_variants["obstacle"] = [
    np.array([0.0, 0.0, 1.8], dtype=np.float32),
    np.array([18.0, 0.0, 5.0], dtype=np.float32),
    np.array([18.0, 0.0, 18.0], dtype=np.float32),
    np.array([65.0, -10.0, 18.0], dtype=np.float32),
    np.array([75.0, -10.0, 5.0], dtype=np.float32)
]

route_plot_filename = f"routes/{algorithm}/{waypoint_variant}/{env_var}_shortest_route_top.png"
route_plot_filename_z = f"routes/{algorithm}/{waypoint_variant}/{env_var}_shortest_route_side.png"

obstacles = [
    np.array([22.0, -22.0, 0.0, 63.0, 20.0, 15.0], dtype=np.float32),
    # np.array([50.0, 0.0, 5.0, 0.0, 0.0, 5.0], dtype=np.float32),
]
if waypoint_variant == "obstacle":
    plot_route_exp([waypoint_variants[waypoint_variant][0] ,waypoint_variants[waypoint_variant][-1]], shortest_routes, obstacles=obstacles, filename=route_plot_filename)
    plot_route_exp_z([waypoint_variants[waypoint_variant][0] ,waypoint_variants[waypoint_variant][-1]], shortest_routes, obstacles=obstacles, filename=route_plot_filename_z)
elif waypoint_variant == "multiple":
    plot_route_exp(waypoint_variants[waypoint_variant], shortest_routes, obstacles=obstacles, filename=route_plot_filename)
else:
    plot_route_exp(waypoint_variants[waypoint_variant], shortest_routes, filename=route_plot_filename)