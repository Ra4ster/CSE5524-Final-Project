import numpy as np
import matplotlib.pyplot as plt
import skimage as sk
from lens1 import otsu

def main():
    # Load the Ti Felt image (the one with the largest error)
    img_raw = sk.io.imread("data/TiFelt_75p_100mm.png", as_gray=True)
    img_uint8 = (img_raw * 255.0).astype(np.uint8) if img_raw.max() <= 1.0 else img_raw.astype(np.uint8)

    # 1. Get Otsu's Threshold
    otsu_t, _ = otsu(img_uint8)
    
    # 2. Get the True 75% Porosity Threshold
    percentile_t = np.percentile(img_uint8, 75.0)

    # 3. Plot the Histogram
    plt.figure(figsize=(10, 5))
    
    # Plot pixel distribution
    plt.hist(img_uint8.ravel(), bins=256, range=(0, 255), color='gray', alpha=0.7)
    
    # Plot Otsu (Red) vs True Target (Green)
    plt.axvline(otsu_t, color='red', linestyle='dashed', linewidth=2, 
                label=f'Otsu Threshold (T={otsu_t:.1f})')
    plt.axvline(percentile_t, color='green', linestyle='dashed', linewidth=2, 
                label=f'True 75% Cutoff (T={percentile_t:.1f})')

    # Fill the "Error Zone" between the two lines
    plt.axvspan(otsu_t, percentile_t, color='red', alpha=0.15, label='Misclassified as Solid')

    plt.title("Histogram Analysis: Ti Felt (75% Target Porosity)", fontweight='bold')
    plt.xlabel("Pixel Intensity (0 = Dark Pore, 255 = Bright Solid)")
    plt.ylabel("Pixel Count")
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/histogram_diagnostic.png", dpi=300)
    print("Saved histogram diagnostic to results/histogram_diagnostic.png")

if __name__ == "__main__":
    main()