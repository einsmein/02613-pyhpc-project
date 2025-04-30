import matplotlib.pyplot as plt
from simulate import load_data

def visualize_data(load_dir, bid):
    # Load data
    u, interior_mask = load_data(load_dir, bid)
    
    # Plot the domain and interior mask
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Visualize the domain
    axes[0].imshow(u, cmap="inferno", origin="lower")
    axes[0].set_title(f"Domain for Building {bid}")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    
    # Visualize the interior mask
    axes[1].imshow(interior_mask, cmap="gray", origin="lower")
    axes[1].set_title(f"Interior Mask for Building {bid}")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")
    
    plt.tight_layout()
    plt.savefig(f"output/visualization_{bid}.png")

if __name__ == "__main__":
    import sys
    building_id = sys.argv[1]
    LOAD_DIR = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"
    visualize_data(LOAD_DIR, building_id)