import matplotlib.pyplot as plt
import numpy as np
import argparse

import os

# this neeeds to be ArgumentParser not Parser
parser = argparse.ArgumentParser(description="file for visualizing files")

# hypehsn are not good use underscore for precicesness
# Additionaly I forgot to consider using "type" parameters as well as the "help parameters" for descrbiging the names
parser.add_argument("--input_dir", "-i", default = "./data/processed_mri_data/", required = False)
parser.add_argument("--patient", '-p', type=int, default = 529, required = False, help = 'patient number')
parser.add_argument('-s', '--slicetype', type=str, default = 'raw', required = False, help='slice type, choose from raw (mask) cine or cine whole')
parser.add_argument("--use-new-directory",action = "store_true", required = False, help='flag for determining whether to visualize new processed images (internal tool)')

args = parser.parse_args()

images = []
name = args.slicetype  

for index in range(5): 
    if args.use_new_directory:
        images.append(np.load(os.path.join(args.input_dir, f'{name}_{index}.npy'))) 
    else:
        images.append(np.load(os.path.join(args.input_dir, f'{args.patient}_PSIR_{name}_{index}.npy'))) 

fig, axes = plt.subplots(1, 5, figsize=(10, 5))

for i, (ax, image) in enumerate(zip(axes, images)) :
    ax.set_title(f"Slice: {i}")
    ax.imshow(image, cmap = 'gray', vmin = 0, vmax = 1)


plt.show()


# Notes: why is there a discrepancy between the cropped and made segmentation? 

