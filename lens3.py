import numpy as np
import scipy
import skimage as sk
from skimage.feature import ORB, match_descriptors
import matplotlib.pyplot as plt
from lens1 import otsu
from lens1_analysis import ptl_dataset
import warnings
import openpyxl as op
from scipy.stats import pearsonr, spearmanr

# Logm prints "may be inaccurate" with small error
warnings.filterwarnings("ignore", category=RuntimeWarning, module="skimage")

def features(img: np.ndarray) -> np.ndarray:
    """
    Extracts features from a 2D grayscale image and computes its covariance matrix.

    Parameters:
        img (np.ndarray): A 2D grayscale image represented as a NumPy array.
    Returns:
        np.ndarray: The covariance matrix of the extracted features.
    """
    r,c = img.shape

    y,x = np.mgrid[0:r, 0:c]

    I = img.astype(float) / 255.
    Ix = sk.filters.sobel_h(I)
    Iy = sk.filters.sobel_v(I)

    Z = np.vstack(( # [x,y,intensity,gradientX,gradientY]
        x.ravel(),
        y.ravel(),
        I.ravel(),
        Ix.ravel(),
        Iy.ravel()
    ))

    C = np.cov(Z)

    epsilon = 1e-6
    C += np.eye(C.shape[0]) * epsilon # Add a small value for positive-definiteness
    return C

def riemann_distance(C_i: np.ndarray, C_j: np.ndarray) -> float:
    """
    Computes the Riemannian distance between two covariance matrices.
    
    Parameters:
        C_i (np.ndarray): The first covariance matrix.
        C_j (np.ndarray): The second covariance matrix.
    Returns:
        float: The Riemannian distance between the two covariance matrices.
    """
    C_i_inv_half = scipy.linalg.fractional_matrix_power(C_i, -0.5)
    core = C_i_inv_half @ C_j @ C_i_inv_half

    log_core = scipy.linalg.logm(core)
    return np.linalg.norm(log_core, 'fro')

def main():
    data = ptl_dataset(xlsx="../ptl_cv_dataset/PTL_CV_Dataset.xlsx")
    
    unique_models = list(dict.fromkeys(data.model_ids))
    
    model_drift_variances = []
    model_porosity_variances = []
    
    print("Running Lens 3 Covariance Tracking (Calculating drifts)...")
    
    # PASS 1: Calculate all the math first
    for current_model in unique_models:
        indices = [i for i, mid in enumerate(data.model_ids) if mid == current_model]
        model_images = [data.images[i] for i in indices]
        model_porosities = [data.local_slice_porosity[i] for i in indices]
        
        covs = [features(img) for img in model_images]
        drifts = [riemann_distance(covs[i], covs[i+1]) for i in range(len(covs)-1)]
        
        model_drift_variances.append(np.var(drifts))
        model_porosity_variances.append(np.var(model_porosities))
        
    # CALCULATION: Use custom Otsu from Lens 1
    drift_vars_array = np.array(model_drift_variances)
    
    # 1. Scale the float variances to 0-255 uint8 integers so np.bincount doesn't crash
    d_min = drift_vars_array.min()
    d_max = drift_vars_array.max()
    
    if d_max > d_min:
        drift_uint8 = ((drift_vars_array - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)
    else:
        drift_uint8 = np.zeros_like(drift_vars_array, dtype=np.uint8)
        
    # 2. Call your custom otsu function
    t_uint8, _ = otsu(drift_uint8)
    
    # 3. Map the uint8 threshold back to the original continuous variance scale
    if d_max > d_min:
        threshold = t_uint8 / 255.0 * (d_max - d_min) + d_min
    else:
        threshold = d_min
        
    print(f"\n--- Custom Otsu Threshold set to: {threshold:.6f} ---")
    
    # PASS 2: Label and print the results
    for i, current_model in enumerate(unique_models):
        drift_variance = model_drift_variances[i]
        porosity_variance = model_porosity_variances[i]
        
        label = "Heterogeneous" if drift_variance > threshold else "Homogeneous"
        print(f"[{current_model}] Drift Variance: {drift_variance:.6f} | Porosity Variance: {porosity_variance:.6f} | Label: {label}")

    # Correlate your visual drift against true porosity variance
    pearson_corr, _ = pearsonr(model_drift_variances, model_porosity_variances)
    spearman_corr, _ = spearmanr(model_drift_variances, model_porosity_variances)
    
    print("\n--- LENS 3 VALIDATION RESULTS ---")
    print(f"Correlation between Covariance Drift and True Porosity Variance: {pearson_corr:.4f}")
    print(f"Spearman Correlation: {spearman_corr:.4f}")
    
    plt.plot(range(1, len(unique_models) + 1), model_drift_variances, marker='o')
    plt.title("Covariance Drift Variance Across All Models (Custom Otsu)")
    plt.xlabel("Model Index")
    plt.ylabel("Drift Variance")
    plt.axhline(y=threshold, color='r', linestyle='--', label=f"Otsu Threshold ({threshold:.6f})")
    plt.legend()
    plt.show()

    saved = input("Would you like to save the drift to Excel? (y/n): ")
    if saved.lower() == 'y':
        wb = op.Workbook(write_only=True)
        ws = wb.create_sheet(title="Lens3 Drift Results")
        ws.append(["Model_ID", "Drift_Variance", "Porosity_Variance"])
        for i, model in enumerate(unique_models):
            ws.append([model, model_drift_variances[i], model_porosity_variances[i]])
        wb.save("Lens3_Drift_Results.xlsx")
    else:
        print("Drift results not saved.")

if __name__ == "__main__":
    main()