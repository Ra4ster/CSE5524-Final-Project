# DEPRECATED
import os

import lens1
import numpy as np
import skimage as sk

elongations = []
porosities = []

os.makedirs("results/Model_TEST", exist_ok=True)

for i in range(15):
    img = sk.io.imread(
        f"Model_TEST/Model_TEST/Model_TEST_img_{i + 1:02d}.png",
        as_gray=True,
    )

    # Connected Components Labeling
    ccl = lens1.ptl_ccl(
        img,
        c=2,
        min_pixels=20,
    )

    # Average pore elongation (eccentricity)
    avg_elongation = ccl.average_elongation()
    elongations.append(avg_elongation)

    # Porosity = pore pixels / total pixels
    porosity = 100.0 * np.mean(img > 0)
    porosities.append(porosity)

    print(
        f"Image {i + 1:02d}: "
        f"{len(ccl.pores())} pores, "
        f"average elongation = {avg_elongation:.4f}, "
        f"porosity = {porosity:.2f}%"
    )

    # Save labeled connected components
    ccl.save(f"results/Model_TEST/img_{i + 1:02d}.png")

print("\n========== Dataset Summary ==========")
print(f"Mean pore elongation: {np.mean(elongations):.4f}")
print(f"Mean porosity:        {np.mean(porosities):.2f}%")