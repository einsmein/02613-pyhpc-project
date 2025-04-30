"""
For debugging, run `python simulate.py 1`.
You can uncomment visualize function call in main to see the plot.
"""

from os.path import join
import sys

from matplotlib import pyplot as plt
import numpy as np

DATA_SIZE = 512
MAX_ITER = 20_000
ABS_TOL = 1e-4

def load_data(load_dir, bid):
    u = np.zeros((DATA_SIZE + 2, DATA_SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = np.sum(u_interior > 18) / u_interior.size * 100
    pct_below_15 = np.sum(u_interior < 15) / u_interior.size * 100
    return {
        "mean_temp": mean_temp,
        "std_temp": std_temp,
        "pct_above_18": pct_above_18,
        "pct_below_15": pct_below_15,
    }


def visualize(all_u):   
    num_matrices = len(all_u)
    fig, axes = plt.subplots(1, num_matrices, figsize=(6 * num_matrices, 6))
    
    if num_matrices == 1:
        axes = [axes]  # Ensure axes is iterable for a single matrix
    
    for i, u in enumerate(all_u):
        axes[i].imshow(u, cmap="inferno", origin="lower")
        axes[i].set_title(f"Matrix {i + 1}")
        axes[i].set_xlabel("X")
        axes[i].set_ylabel("Y")
    
    plt.tight_layout()
    plt.savefig("output/visualization.png")


def check_result_is_close(u0, interior_mask, u):
    og_u = og_jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)
    is_close = np.allclose(og_u,u)
    return is_close


def og_jacobi(u, interior_mask, max_iter, atol=1e-6):
    u = np.copy(u)

    for i in range(max_iter):
        # Compute average of left, right, up and down neighbors, see eq. (1)
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior

        if delta < atol:
            break
    return u

#######################################################
### TODO: Optimize `simulate` and `jacobi`
#######################################################

def jacobi(u, interior_mask, max_iter, atol=1e-6):
    u = np.copy(u)

    for i in range(max_iter):
        # Compute average of left, right, up and down neighbors, see eq. (1)
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior

        if delta < atol:
            break
    return u


def simulate(all_u0, all_interior_mask):
    all_u = np.empty_like(all_u0)
    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        u = jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)
        all_u[i] = u
    return all_u


####################################################
# Main function
####################################################
if __name__ == "__main__":
    # Load data
    LOAD_DIR = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"
    with open(join(LOAD_DIR, "building_ids.txt"), "r") as f:
        building_ids = f.read().splitlines()

    if len(sys.argv) < 2:
        N = 1
    else:
        N = int(sys.argv[1])
    building_ids = building_ids[:N]

    # Load floor plans
    all_u0 = np.empty((N, 514, 514))                            # All initial temp
    all_interior_mask = np.empty((N, 512, 512), dtype="bool")   # All interior mask
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    # Run jacobi iterations for each floor plan
    all_u = simulate(all_u0, all_interior_mask)

    # For debugging, visualize the results
    visualize(all_u)

    # For checking correctness, compare the first result with the original Jacobi method
    print("Test")
    print("----------------")
    print("Checking the first result against the original Jacobi method...")
    is_close = check_result_is_close(all_u0[0], all_interior_mask[0], all_u[0])
    if is_close is False:
        print("Your result is incorrect")
        print("!!!! FIX YOUR CODE BEFORE MOVING ON !!!!")
    else:
        print("Your result is correct")
    print()

    # Print summary statistics in CSV format
    print("Result summary")
    print("----------------")
    stat_keys = ["mean_temp", "std_temp", "pct_above_18", "pct_below_15"]
    print("building_id, " + ", ".join(stat_keys))  # CSV header
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))
