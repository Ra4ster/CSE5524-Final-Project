import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import skimage as sk
from scipy.ndimage import center_of_mass, gaussian_filter, label
import time
import math
from typing import Optional

IMG1_1: np.ndarray = sk.io.imread('data/ThickTiSinter_41p_270t_500mm.png')
IMG1_2: np.ndarray = sk.io.imread('data/ThickTiSinter_41p_270t_500mm.png')
IMG2_1: np.ndarray = sk.io.imread('data/ThinTiSinter_52p_40t_100mm.png')
IMG2_2: np.ndarray = sk.io.imread('data/ThinTiSinter_52p_40t_500mm.png')
IMG3_1: np.ndarray = sk.io.imread('data/TiFelt_75p_100mm.png')
IMG3_2: np.ndarray = sk.io.imread('data/TiFelt_75p_500mm.png')

# Part 1: Thresholding methods

def otsu(img: np.ndarray) -> tuple[int, float]:
    """
    Computes the optimal binary threshold of a grayscale image using
    Otsu's method.

    Parameters:
        img (np.ndarray):
            A grayscale image represented as a 2-D `uint8` NumPy array.

    Returns:
        tuple[int, float]:
            thresh (int):
                The optimal threshold in the range [0, 255].
                For binary images, this is the non-zero pixel value (e.g., 255 or 1).
            var (float):
                The maximum between-class variance.
    """
    
    flat = img.ravel()
    unique = np.unique(flat)
    if len(unique) == 2: # Binary image
        return (128, 0.) # Just return middle
    sz = np.size(flat)
    thresh = 0 ; var = float("-inf")

    hist = np.bincount(flat, minlength=256)
    p = hist / hist.sum()

    omega = np.cumsum(p)
    mu = np.cumsum(np.arange(256) * p)
    mu_total = mu[-1]

    for t in range(0, 256, 1):
        w0 = omega[t]
        w1 = 1 - w0

        if w0 == 0 or w1 == 0:
            continue

        mu0 = mu[t] / w0
        mu1 = (mu_total - mu[t]) / w1

        s2 = w0 * w1 * (mu0 - mu1) ** 2
        if s2 > var:
            thresh = t
            var = s2

    return (thresh, var)

def multi_otsu(img: np.ndarray) -> tuple[int, int]:
    """
    Custom 3-class Multi-Otsu implementation.
    Returns two thresholds (t1, t2) to separate the image into:
      Class 0: 0 to t1 (Void)
      Class 1: t1+1 to t2 (Boundary Shadow)
      Class 2: t2+1 to 255 (Solid Fiber)
    """
    hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 256))
    prob = hist.astype(float) / img.size

    omega = np.cumsum(prob) 
    mu = np.cumsum(prob * np.arange(256))
    mu_total = mu[-1] # Global mean of the entire image

    max_variance = -1.0
    t1_opt, t2_opt = 0, 0

    # Iterate through all possible threshold pairs (t1 < t2)
    for t1 in range(1, 254):
        w0 = omega[t1]
        if w0 == 0: 
            continue

        for t2 in range(t1 + 1, 255):
            w1 = omega[t2] - omega[t1]
            if w1 == 0: 
                continue
            
            w2 = omega[255] - omega[t2]
            if w2 == 0: 
                continue

            mu0 = mu[t1] / w0
            mu1 = (mu[t2] - mu[t1]) / w1
            mu2 = (mu[255] - mu[t2]) / w2

            variance = (w0 * (mu0 - mu_total)**2 + 
                        w1 * (mu1 - mu_total)**2 + 
                        w2 * (mu2 - mu_total)**2)

            if variance > max_variance:
                max_variance = variance
                t1_opt = t1
                t2_opt = t2

    return t1_opt, t2_opt

# --- Example usage: ---

# start = time.perf_counter()
# thresh, _ = otsu(img)
# elapsed = time.perf_counter() - start

# hours, remainder = divmod(elapsed, 3600)
# minutes, seconds = divmod(remainder, 60)

# print(f"Optimal threshold: {thresh}")
# print(f"Time taken: {seconds:.5f} seconds")

# mask = img < thresh

# plt.imshow(mask, cmap="gray")
# plt.show()

# Part 2: Thresholding evaluation

def IoU(pred: np.ndarray, target: np.ndarray) -> float:
    assert pred.dtype == bool
    assert target.dtype == bool

    num: float = np.sum(pred & target)
    den: float = np.sum(pred | target)
    return num / den

def Dice(pred: np.ndarray, target: np.ndarray) -> float:
    assert pred.dtype == bool
    assert target.dtype == bool

    num: float = 2.0 * np.sum(pred & target)
    den: float = np.sum(pred) + np.sum(target)
    return num / den

class ptl_ccl:
    """
    Porous Transport Layer's Connected Components Labeling (PTL-CCL)
    """

    regions = []
    labels: Optional[np.ndarray] = None

    def __init__(self, img: np.ndarray, c: int = 2, min_pixels: int = 20):
        self.labels = sk.measure.label(img, connectivity = c, return_num=False)
        self.regions = [
            r for r in sk.measure.regionprops(self.labels)
            if r.area >= min_pixels
        ]

    def pores(self) -> list:
        return self.regions
    
    def eccentricity(self, i: int) -> float:
        return self.regions[i].eccentricity
    
    def area(self, i: int) -> float:
        return self.regions[i].area
    
    def perimeter(self, i: int) -> float:
        return self.regions[i].perimeter
    
    def moments_hu(self, i: int) -> float:
        return self.regions[i].moments_hu
    
    def compactness(self, i: int) -> float:
        return (4 * math.pi * self.area(i)) / (self.perimeter(i) ** 2)
    
    def average_elongation(self) -> float:
        if not self.pores():
            return 0.0
        
        total_eccentricity = 0.0
        for i in range(len(self.pores())):
            total_eccentricity += self.eccentricity(i)

        return total_eccentricity / len(self.pores())

    def _render_plot(self, max_labels: int = 20) -> tuple[plt.Figure, plt.Axes]:
        img_overlay = sk.color.label2rgb(self.labels, bg_label=0)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(img_overlay)

        # Sort regions by area (descending) so we only label the top N largest
        sorted_regions = sorted(enumerate(self.regions), key=lambda x: x[1].area, reverse=True)
        top_regions = sorted_regions[:max_labels]

        for idx, pore in top_regions:
            y0, x0 = pore.centroid
            ax.text(
                x0, y0, str(idx), 
                color='white', fontsize=8, fontweight='bold', 
                ha='center', va='center',
                bbox=dict(boxstyle='circle,pad=0.15', facecolor='black', alpha=0.6, edgecolor='none')
            )

        ax.set_axis_off()
        plt.tight_layout()
        return fig, ax

    def show(self, max_labels: int = 20) -> None:
        fig, _ = self._render_plot(max_labels=max_labels)
        plt.show()

    def save(self, filename: str, max_labels: int = 20) -> None:
        fig, _ = self._render_plot(max_labels=max_labels)
        fig.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close(fig)

# print("Now making Connected Components; this may take a bit: ")
# start = time.perf_counter()

# img_regions = ptl_ccl(mask, c=2, min_pixels=20)

# elapsed = time.perf_counter() - start
# hours, remainder = divmod(elapsed, 3600)
# minutes, seconds = divmod(remainder, 60)
# print(f"Connected components completed (took {seconds:.5f} seconds).")
# print(f"There are {len(img_regions.regions)} components.")
# avg_elongation = img_regions.average_elongation()
# print(f"Average Pore Elongation (Eccentricity): {avg_elongation:.4f}")
# print("Now showing regions!")
# img_regions.show()

def harris_corners(img: np.ndarray, k: float = 0.01, sigma: float = 1., t: float = 0.01, m: float =0) -> np.ndarray:
    Ix = sk.filters.sobel_h(img)
    Iy = sk.filters.sobel_v(img)

    A = gaussian_filter(Ix * Ix, sigma)
    B = gaussian_filter(Ix * Iy, sigma)
    C = gaussian_filter(Iy * Iy, sigma)

    d = A*C - B**2
    trace = A + C

    response = d - k * (trace ** 2)

    threshold = t * response.max()
    mask = response > threshold
    return sk.feature.peak_local_max(response, min_distance=m, threshold_abs=threshold)

def plot_harris_corners(img: np.ndarray, corners: np.ndarray, save_path: str = "results/harris_corners_full.png") -> None:
    """
    Overlays detected Harris corners across the entire image.
    
    Parameters:
      img: 2D numpy array (grayscale image)
      corners: Array of corner coordinates, typically shape (N, 2) [row, col] or [y, x]
      save_path: Output file path
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Display the base image in grayscale
    ax.imshow(img, cmap='gray')
    
    # Extract x and y coordinates
    # Note: harris_corners typically returns [row, col] -> [y, x]
    if len(corners) > 0:
        y_coords, x_coords = corners[:, 0], corners[:, 1]
        
        # Plot red dots/crosses over detected corner points across the entire image
        ax.scatter(
            x_coords, y_coords, 
            s=15, c='#e7298a', marker='o', edgecolors='black', linewidths=0.5,
            label=f'Corners Detected ({len(corners)})'
        )
    
    ax.set_title(f"Harris Corner Detection (Total: {len(corners)})", fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[RESULT] Saved full-image Harris corner plot to {save_path}")

# print("Computing harris corner points of original image.")
# start = time.perf_counter()

# hc = harris_corners(img.astype(float) / 255.0)

# elapsed = time.perf_counter() - start
# hours, remainder = divmod(elapsed, 3600)
# minutes, seconds = divmod(remainder, 60)
# print(f"Harris corners calculated (there are {hc.size}). Time taken: {seconds:.5f} seconds.")

# plt.imshow(img, cmap="gray")
# plt.scatter(hc[:,1], hc[:,0],
#             c="red", s=20)
# plt.show()

def main():
    img = (sk.io.imread('real.png', as_gray=True) * 255.0).astype(float)

    corners = harris_corners(img / 255.0, t=0.03, m=0)

    plot_harris_corners(img, corners, save_path="results/lens1_harris_corners_full.png")


if __name__ == "__main__":
    main()