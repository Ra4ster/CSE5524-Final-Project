import lens1
import numpy as np
import skimage as sk

averages: list = []
for i in range(15):
    img = sk.io.imread(
        f'Model_TEST/Model_TEST/Model_TEST_img_{i + 1:02d}.png',
        as_gray=True
    )

    ccl = lens1.ptl_ccl(img, c=2, min_pixels=20)

    averages.append(ccl.average_elongation())

    print(
        f"Image {i+1:02d}: "
        f"{len(ccl.pores())} pores, "
        f"average elongation = {averages[i]:.4f}"
    )
    ccl.save(f'results/Model_TEST/img_{i + 1:02d}.png')