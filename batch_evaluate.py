import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import skimage as sk
from lens1 import otsu, ptl_ccl, harris_corners

def main():
    os.makedirs("results", exist_ok=True)

    samples = [
        {"name": "Thick Ti Sinter (270t, 100mm)", "file": "data/ThickTISinter_41p_270t_100mm.png", "target_p": 0.41},
        {"name": "Thick Ti Sinter (270t, 500mm)", "file": "data/ThickTiSinter_41p_270t_500mm.png", "target_p": 0.41},
        {"name": "Thin Ti Sinter (40t, 100mm)", "file": "data/ThinTiSinter_52p_40t_100mm.png", "target_p": 0.52},
        {"name": "Thin Ti Sinter (40t, 500mm)", "file": "data/ThinTiSinter_52p_40t_500mm.png", "target_p": 0.52},
        {"name": "Ti Felt (100mm)", "file": "data/TiFelt_75p_100mm.png", "target_p": 0.75},
        {"name": "Ti Felt (500mm)", "file": "data/TiFelt_75p_500mm.png", "target_p": 0.75},
    ]

    results_summary = []
    fig_master, axes_master = plt.subplots(2, 3, figsize=(15, 9))
    axes_flat = axes_master.flatten()

    print("Processing Lens 1 across all 6 samples...\n" + "="*50)

    for idx, sample in enumerate(samples):
       # 1. Load Image
        img_raw = sk.io.imread(sample["file"], as_gray=True)
        img_uint8 = (img_raw * 255.0).astype(np.uint8) if img_raw.max() <= 1.0 else img_raw.astype(np.uint8)

        # 2. Standard Otsu (Using your custom function from lens1.py)
        t, _ = otsu(img_uint8)
        
        # Pores are the dark regions (pixel values below the threshold)
        pore_mask = img_uint8 < t
        measured_p = np.mean(pore_mask)

        # 3. Connected Components
        ccl = ptl_ccl(pore_mask, c=2, min_pixels=20)
        ccl.show()
        # 4. Harris Corners
        img_float = img_uint8.astype(float) / 255.0
        corners = harris_corners(img_float, t=0.03, m=3)

        # Store metrics
        results_summary.append({
            "Name": sample["name"],
            "Target": f"{sample['target_p']*100:.0f}%",
            "Measured": f"{measured_p*100:.1f}%",
            "Regions": len(ccl.pores()),
            "Elongation": f"{ccl.average_elongation():.3f}",
            "Corners": len(corners)
        })

        # Render panel
        ax = axes_flat[idx]
        ax.imshow(img_uint8, cmap='gray')
        if len(corners) > 0:
            ax.scatter(corners[:, 1], corners[:, 0], c='#e7298a', s=12, edgecolors='black', linewidths=0.3)
        ax.set_title(f"{sample['name']}\nMeas P: {measured_p*100:.1f}% | Corners: {len(corners)}", fontsize=9, fontweight='bold')
        ax.axis('off')

    # Save output
    plt.tight_layout()
    fig_master.savefig("results/master_lens1_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig_master)

    # Print table
    print(f"{'Sample Name':<32} | {'Target P':<8} | {'Meas P':<8} | {'Regions':<8} | {'Elongation':<10} | {'Corners':<8}")
    print("-" * 90)
    for r in results_summary:
        print(f"{r['Name']:<32} | {r['Target']:<8} | {r['Measured']:<8} | {r['Regions']:<8} | {r['Elongation']:<10} | {r['Corners']:<8}")

if __name__ == "__main__":
    main()