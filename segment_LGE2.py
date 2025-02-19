"""

New and improved segmenter to test the dataset object for visualizing.
"""

import matplotlib.pyplot as plt
from attention_model import AttentionUNet
from dataset import LGEDataset, Preprocessor
import torch

DEVICE = "cuda:0"

pp = Preprocessor()


X, y, ids = pp.transform() # this thing takes the image directories and works 
dataset = LGEDataset(X, y, ids)


model_save_path= './models/unet_att_focal_dice350.pt'

model = AttentionUNet(drop_out_prob=0.3).to(DEVICE)
model.load_state_dict(torch.load('./' + model_save_path))
model.eval()


images, label, patient_id = dataset[109]

raw_image = images[2][0].unsqueeze(0).unsqueeze(0)

y_pred = model(raw_image.to(DEVICE))

# Process the image logits with thresholded binary values
binary_thresholded_y = y_pred.detach().cpu().numpy().squeeze(0).squeeze(0)

threshold = 0.5 
binary_thresholded_y[binary_thresholded_y < threshold] = 0 
binary_thresholded_y[binary_thresholded_y >= threshold] = 255 # the images are scaled from 0 255
# deatch().cpu().numpy() === get rid of gradietn computation (needed for getting using as numpy on cpu)

fig, axes = plt.subplots(1, 4, figsize=(10, 5))

import numpy as np

cine_cropped = np.load(f'./data/processed_mri_data/{patient_id}_PSIR_cine_0.npy')
cine_whole= np.load(f'./data/processed_mri_data/{patient_id}_PSIR_cine_whole_0.npy')

raw = np.load(f'./data/processed_mri_data/{patient_id}_PSIR_raw_0.npy')
axes[0].imshow(cine_whole, cmap = 'gray')

axes[1].imshow(cine_cropped, cmap = 'gray')
axes[2].imshow(raw, cmap = 'gray')
axes[3].imshow(binary_thresholded_y)


plt.show()




