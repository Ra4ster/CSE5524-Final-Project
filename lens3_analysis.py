import numpy as np
from sklearn.linear_model import LinearRegression
from lens3 import features, riemann_distance
import scipy
from scipy.stats import rankdata
import skimage as sk
from lens1 import otsu, ptl_ccl
from lens1_analysis import ptl_dataset
import openpyxl as op


def xy_pore_variance(img: np.ndarray) -> float:
    thresh,_ = otsu(img)
    mask = img > thresh

    img_regions = ptl_ccl(mask, c=2, min_pixels=20)

    areas = np.array([img_regions.area(i) for i in range(len(img_regions.pores()))])
    if len(areas) > 0:
        return np.var(areas)
    else: return 0.0

def main():
    print("[DEBUG] Loading Dataset...")
    data = ptl_dataset(xlsx="../ptl_cv_dataset/PTL_CV_Dataset.xlsx")

    unique_models = list(dict.fromkeys(data.model_ids))
    model_z_drift = []
    model_xy_variances = []
    model_tortuosities = []
    model_porosity_variances = []

    print("[DEBUG] Running Unified Lens 1 & 3 Tracking...")

    for current_model in unique_models:
        indices = [i for i, mid in enumerate(data.model_ids) if mid == current_model]
        model_images = [data.images[i] for i in indices]
        model_porosities = [data.local_slice_porosity[i] for i in indices]
        
        # --- Z-Axis: Covariance Drift ---
        covs = [features(img) for img in model_images]
        z_drifts = [riemann_distance(covs[i], covs[i+1]) for i in range(len(covs)-1)]
        model_z_drift.append(np.var(z_drifts))
        
        # --- XY-Axis: Lens 1 CCL Pore Area Variance ---
        xy_vars = [xy_pore_variance(img) for img in model_images]
        model_xy_variances.append(np.mean(xy_vars))

        # --- Tortuosity: Lens 1 Elongation ---
        elongations = []
        for img in model_images:
            t, _ = otsu(img)
            mask = img < t
            ccl = ptl_ccl(mask, c=2, min_pixels=20)
            elongations.append(ccl.average_elongation())
        model_tortuosities.append(np.mean(elongations))
        
        # --- Ground Truth ---
        model_porosity_variances.append(np.var(model_porosities))

    pearson_z,_ = scipy.stats.pearsonr(model_z_drift, model_porosity_variances)
    spearman_z,_ = scipy.stats.spearmanr(model_z_drift, model_porosity_variances)

    pearson_xy,_ = scipy.stats.pearsonr(model_xy_variances, model_porosity_variances)
    spearman_xy,_ = scipy.stats.spearmanr(model_xy_variances, model_porosity_variances)

    print("\n--- Z-Axis (Inter-slice Covariance Drift) ---")
    print(f"Pearson Correlation: {pearson_z}")
    print(f"Spearman Correlation: {spearman_z}")

    print("\n--- XY-Axis (Intra-slice Pore Area Variance) ---")
    print(f"Pearson Correlation: {pearson_xy}")
    print(f"Spearman Correlation: {spearman_xy}")
    
    z_ranks = rankdata(model_z_drift)
    xy_ranks = rankdata(model_xy_variances)
    tort_ranks = rankdata(model_tortuosities)
    y_ranks = rankdata(model_porosity_variances)

    X_ranked = np.column_stack((z_ranks, xy_ranks, tort_ranks))

    reg = LinearRegression().fit(X_ranked,y_ranks)
    combined_ranked_pred = reg.predict(X_ranked)

    combined_spearman,_ = scipy.stats.spearmanr(combined_ranked_pred, y_ranks)

    print("\n--- Combined 3D Diagnostic (Z + XY + Tortuosity) ---")
    print(f"Spearman Correlation: {combined_spearman}")
    
    print("\n--- Feature Weights ---")
    print(f"Z-Drift:    {reg.coef_[0]:.4f}")
    print(f"XY-Var:     {reg.coef_[1]:.4f}")
    print(f"Tortuosity: {reg.coef_[2]:.4f}")

    saved = input("\nWould you like to save the unified results to Excel? (y/n): ")
    if saved.lower() == 'y':
        wb = op.Workbook(write_only=True)
        ws = wb.create_sheet(title="Unified 3D Diagnostics")

        headers = [
            "Model_ID", 
            "Z_Drift_Variance", 
            "XY_Pore_Variance", 
            "Average_Tortuosity", 
            "True_Porosity_Variance"
        ]
        ws.append(headers)

        for i in range(len(unique_models)):
            row_data = [
                unique_models[i],
                model_z_drift[i],
                model_xy_variances[i],
                model_tortuosities[i],
                model_porosity_variances[i]
            ]
            ws.append(row_data)

        output_file_name = "Unified_Lens3_Results.xlsx"
        wb.save(output_file_name)
        print(f"[RESULT] Successfully saved to {output_file_name}")


    else:
        print("Results not saved.")

if __name__ == "__main__":
    main()