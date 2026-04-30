import os
import cv2
import numpy as np

blurry_dir = "Deraining/testsets/Test2800_small"
restored_dir = "results/Test2800_small/Deraining"
save_dir = "results/Test2800_small/comparisons"

os.makedirs(save_dir, exist_ok=True)

for filename in os.listdir(blurry_dir):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        blurry_path = os.path.join(blurry_dir, filename)
        restored_path = os.path.join(restored_dir, filename)

        if not os.path.exists(restored_path):
            print(f"Skipping {filename}: no restored version found.")
            continue

        blurry = cv2.imread(blurry_path)
        restored = cv2.imread(restored_path)

        # Resize restored to match blurry if needed
        if blurry.shape != restored.shape:
            restored = cv2.resize(restored, (blurry.shape[1], blurry.shape[0]))

        comparison = np.hstack((blurry, restored))
        save_path = os.path.join(save_dir, filename)

        cv2.imwrite(save_path, comparison)
        print(f"Saved comparison: {save_path}")

