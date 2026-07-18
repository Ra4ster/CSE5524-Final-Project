import lens1
import numpy as np
import skimage as sk
import os

averages: list = []

os.makedirs("results/Model_TEST", exist_ok=True)

for i in range(15):
    img = sk.io.imread(
        f'Model_TEST/Model_TEST/Model_TEST_img_{i + 1:02d}.png',
        as_gray=True
    )

    # Check image values
    print(
        f"Image {i+1:02d}: "
        f"min={img.min():.3f}, "
        f"max={img.max():.3f}, "
        f"unique={len(np.unique(img))}"
    )

    # Use existing segmentation directly
    ccl = lens1.ptl_ccl(
        img,
        c=2,
        min_pixels=20
    )

    avg_elongation = ccl.average_elongation()
    averages.append(avg_elongation)

    # Estimate porosity from the non-background pixels
    porosity_percent = 100.0 * np.mean(img > 0)

    print(
        f"Image {i+1:02d}: "
        f"{len(ccl.pores())} pores, "
        f"average elongation = {avg_elongation:.4f}, "
        f"porosity = {porosity_percent:.2f}%"
    )

    ccl.save(
        f'results/Model_TEST/img_{i + 1:02d}.png'
    )

print(f"\nDataset average elongation: {np.mean(averages):.4f}")