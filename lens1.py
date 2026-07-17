import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import skimage as sk
from skimage.measure._regionprops import RegionProperties as RegionProps
from scipy.ndimage import gaussian_filter
import time
import math

img: np.ndarray = sk.io.imread('./sample_data/0006.png', as_gray=True)
img = (img * 255.0).astype(np.uint8)

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
            var (float):
                The maximum between-class variance.
    """
    
    flat = img.ravel()
    sz = np.size(flat)
    thresh = 0 ; var = float("-inf")

    hist = np.bincount(flat, minlength=256)
    p = hist / hist.sum()
    omega = np.cumsum(p)

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

# --- Example usage: ---

start = time.perf_counter()
thresh, _ = otsu(img)
elapsed = time.perf_counter() - start

hours, remainder = divmod(elapsed, 3600)
minutes, seconds = divmod(remainder, 60)
print("\nOtsu's method testing... \n")
print(f"Optimal threshold: {thresh}")
print(f"Time taken: {seconds:.5f} seconds\n")

mask = img < thresh

plt.imshow(mask, cmap="gray")
plt.show()

# Porosity:

void_pixels = np.count_nonzero(mask)
total_pixels = mask.size

print("Fetching porosity...\n")
porosity = void_pixels / total_pixels
print(f"Total void pixels: {void_pixels}")
print(f"Total pixels: {total_pixels}")
print(f"2D Porosity: {porosity:.2f}%\n")

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

    regions: list[RegionProps] = None
    labels: np.ndarray = None

    def __init__(self, img: np.ndarray, c: int = 2, min_pixels: int = 20):
        self.labels = sk.measure.label(img, connectivity = c)
        self.regions = [
            r for r in sk.measure.regionprops(self.labels)
            if r.area >= min_pixels
        ]

    def pores(self) -> list[RegionProps]:
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
    
    def show(self) -> None:
        img_overlay = sk.color.label2rgb(self.labels, bg_label=0)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(img_overlay)

        for i, pore in enumerate(self.regions):
            y0, x0 = pore.centroid
            ax.text(x0, y0, str(i), color='white', fontsize=8, ha='center', va='center')
        ax.set_axis_off()
        plt.tight_layout()
        plt.show()

print("Now making Connected Components; this may take a bit: ")
start = time.perf_counter()

img_regions = ptl_ccl(mask, c=2, min_pixels=20)

elapsed = time.perf_counter() - start
hours, remainder = divmod(elapsed, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"Connected components completed (took {seconds:.5f} seconds).")
print(f"There are {len(img_regions.regions)} components.")
avg_elongation = img_regions.average_elongation()
print(f"Average Pore Elongation (Eccentricity): {avg_elongation:.4f}")
print("\nNow showing regions!")
img_regions.show()

def harris_corners(img: np.ndarray, k: float = 0.05, sigma: float = 1.0, t: float = 0.05) -> np.ndarray:
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
    return sk.feature.peak_local_max(response, min_distance=3, threshold_abs=threshold)

print("Computing harris corner points of original image.")
start = time.perf_counter()

hc = harris_corners(img.astype(float) / 255.0)

elapsed = time.perf_counter() - start
hours, remainder = divmod(elapsed, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"Harris corners calculated (there are {hc.size}). Time taken: {seconds:.5f} seconds.")

plt.imshow(img, cmap="gray")
plt.scatter(hc[:,1], hc[:,0],
            c="red", s=20)
plt.show()

