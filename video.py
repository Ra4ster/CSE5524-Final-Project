# REQUIRES FFMPEG!

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import warnings
from lens1_analysis import ptl_dataset
from lens1 import otsu, ptl_ccl
from lens3 import features, riemann_distance

warnings.filterwarnings("ignore")

def main():
    print("Loading dataset...")
    data = ptl_dataset(xlsx="../ptl_cv_dataset/PTL_CV_Dataset.xlsx")
    
    # Pick a highly heterogeneous model so the visual shifts are obvious
    target_model = "Model_0028"
    print(f"Extracting data for {target_model}...")
    
    indices = [i for i, mid in enumerate(data.model_ids) if mid == target_model]
    images = [data.images[i] for i in indices]
    
    # Pre-calculate metrics to keep the animation fast
    print("Pre-calculating metrics (this will take a minute)...")
    covs = [features(img) for img in images]
    z_drifts = [0.0] + [riemann_distance(covs[i], covs[i+1]) for i in range(len(covs)-1)]
    
    porosities = []
    elongations = []
    for img in images:
        t, _ = otsu(img)
        porosities.append(np.sum(img > t) / img.size)
        
        mask = img < t
        ccl = ptl_ccl(mask, c=2, min_pixels=20)
        elongations.append(ccl.average_elongation())

    # --- Setup the Figure ---
    fig, (ax_img, ax_graph) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Z-Stack Diagnostic: {target_model}", fontsize=16)

    # Left Panel: Image Slice
    img_display = ax_img.imshow(images[0], cmap="gray")
    ax_img.axis("off")
    ax_img.set_title("Current Slice", fontsize=12)

    # Right Panel: Live Drift Graph & Metrics
    ax_graph.set_xlim(0, len(images))
    ax_graph.set_ylim(0, max(z_drifts) * 1.2)
    ax_graph.set_title("Riemannian Z-Drift", fontsize=12)
    ax_graph.set_xlabel("Slice Index")
    ax_graph.set_ylabel("Drift Distance")
    
    line, = ax_graph.plot([], [], lw=2, color='blue')
    dot, = ax_graph.plot([], [], 'ro') # Red dot for current slice
    
    # Text box for live metrics
    metric_text = ax_graph.text(0.05, 0.95, "", transform=ax_graph.transAxes, 
                                fontsize=11, verticalalignment='top', 
                                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    def init():
        line.set_data([], [])
        dot.set_data([], [])
        metric_text.set_text("")
        return img_display, line, dot, metric_text

    def update(frame):
        # Update Image
        img_display.set_array(images[frame])
        
        # Update Graph
        x_data = list(range(frame + 1))
        y_data = z_drifts[:frame + 1]
        line.set_data(x_data, y_data)
        dot.set_data([frame], [z_drifts[frame]])
        
        # Update Text
        text = (f"Slice: {frame + 1} / {len(images)}\n"
                f"Porosity: {porosities[frame]:.4f}\n"
                f"Elongation: {elongations[frame]:.4f}\n"
                f"Slice Z-Drift: {z_drifts[frame]:.4f}")
        metric_text.set_text(text)
        
        return img_display, line, dot, metric_text

    print("Generating animation...")
    ani = animation.FuncAnimation(fig, update, frames=len(images), 
                                  init_func=init, blit=True, interval=200)

    # Save as MP4 using FFmpeg
    output_filename = f"{target_model}_Visualization.mp4"
    writer = animation.FFMpegWriter(fps=5, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(output_filename, writer=writer)
    print(f"Saved successfully to {output_filename}")

if __name__ == "__main__":
    main()