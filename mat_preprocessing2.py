import scipy.io as sio
import scipy
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import tomni
from matplotlib import cm
import torch
import json
import argparse
import pdb


from tqdm import tqdm
"""
Modifications from original 


"""
def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for Preprocessing .mat PSIR/LGE/MAG files from flat directory, this will generate the npy files and cropped version from mat data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '-i', '--input-dir',
        type=str,
        required=False,
        default='C:/Users/jonathanle/downloads/lge_data/data',
        help='Path to directory containing .mat files'
    )
    parser.add_argument("--save_json", action = 'store_true', required = False, help = 'save all the files to json format')
 
    parser.add_argument(
        '-o', '--output_dir',
        type=str,
        required=False,
        default='./data/processed_mri_data',
        help='Path to directory to output processed files'
    )
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_dir):
        parser.error(f'Input directory {args.input_dir} does not exist')
    
    return args
"""
def std_img(tens, global_min = 0, global_max = 255):
    # this is equivalent to dividing by 255

    t_ = (tens- global_min) /(global_max - global_min)
    return t_
"""

# what pixel values do you want to emphasize and why si the scaling important?
def std_img(tens, min_p_value = None, max_p_value = None):
    # amin and amax are same as the max and min versions

    min_value = np.amin(tens)
    max_value = np.amax(tens)

    # introduce a custom scaling option for normalization
    if min_p_value is not None and max_p_value is not None: 
        min_value = min_p_value
        max_value = max_p_value


    result = tens.copy()
    
    # Use np.maximum to ensure we don't go below 0 added this because normalizing myo mask requires this
    result = np.maximum((result.astype(np.float64) - min_value) / (max_value - min_value), 0)


    #normalized_values = (tens-min_value)/(max_value - min_value)
    return result


def resize_volume(img, ex=128):
    # this linear interpolation introduces subtle effects on the volume

    # Why is this importnat? 

    """Resize 2D image to expected size"""
    current_depth = img.shape[0]
    current_width = img.shape[1]            
    depth_factor = ex / current_depth
    width_factor = ex / current_width
    factors = (depth_factor, width_factor)
    return scipy.ndimage.zoom(img, factors, order=1)

def process_mat_file(file_path, saving_folder, filename):
    """Process a single .mat file and save its processed slices"""
    # raw image shape  (1, slice_no, height, width)
    
    # Create output directory if it doesn't exist
    os.makedirs(saving_folder, exist_ok=True)


    def load_files(file_path): 
        # Load the .mat file
        try:
            data = sio.loadmat(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return
        # Verify it's cardiac data
        try:
            assert data['series_type'] == np.array(['Myocardial Evaluation'])
        except:
            print(f"File {filename} is not a cardiac evaluation sequence")
            return

        return data
            
    data = load_files(file_path)
    
    if data is None: 
        raise Exception("error no filepath or file is not myocardial evluation sequence")

    print(f'Processing {filename} - Number of slices: {data["enhancement"][0].shape[0]}')





    # Process each slice
    for slice_no in range(data['enhancement'][0].shape[0]):
        # Get enhancement data (scar)
        scar = np.copy(data['enhancement'][0][slice_no]).astype('float')
        scar[scar == 0] = np.nan


        # Verify endo slice
        try:
            _ = data['lv_endo'][0][slice_no][0][0]
        except:
            print(f'Could not get lv_endo for slice {slice_no}')
            continue
            
        # Get image shape and create masks
        img_shape = np.transpose(data['raw_image'][0,slice_no]).shape

        # Create myo mask
        try:
            myo_seg_endo = tomni.make_mask.make_mask_contour(
                img_shape, data['lv_endo'][0][slice_no][0][0][0])
            myo_seg_epi = tomni.make_mask.make_mask_contour(
                # non homogenous array of what format? 
                img_shape, data['lv_epi'][0][slice_no][0][0][0]) #  data['lv_epi'] shape: (table, slice_no, point, dimension) --> (1, num_slices, # points, 2)
                # get the contours
        except: 
            print(f'error could not get lv epi or endo discarding slice')
            continue 
        # Create myocardium mask

        myo_seg = (myo_seg_epi - myo_seg_endo).astype('float')


        # normalize image by finding the max NOTE: amax == max for numpy
        data['raw_image'][0,slice_no] /= np.amax(data['raw_image'][0,slice_no]) # uses np standard indexing to get an image, 


        myo_seg[myo_seg == 0] = np.nan
        
        # Create final images
        fin_img = data['raw_image'][0,slice_no] * myo_seg # for the slice create the segmented versio
        imc_ = data['raw_image'][0,slice_no] # for the slice, get the originall normalized version




        # normalize something alreaedy normalized? --> when i am normalizing the relative values stay the same when normalizej 

        imc_full = std_img(np.array(data['raw_image'][0,slice_no])) # what effect does this have on normalized values? 


        fin_img[np.isnan(fin_img)] = 0
        
        # Convert to PIL images - what scale is this? do scales matter?
        im = Image.fromarray(np.uint8(cm.gray(fin_img)*255)).convert('L') # why is L importnat? 
        imc__ = Image.fromarray(np.uint8(cm.gray(imc_)*255)).convert('L') # Create unnormalized versions
        scar_im = Image.fromarray(np.uint8(cm.gray(scar)*255)).convert('L')


        if os.path.splitext(filename)[0] == '529_PSIR' and slice_no == 1:
            pass
            #pdb.set_trace()
        # normalize the whole image --> crop out the image, 

        # Introduce a standarized normalization process 
            
        # Crop and resize
        bbox = im.getbbox()
        cropped_unormalized_image = np.array(imc__.crop(bbox))

        imc = std_img(cropped_unormalized_image)  # cropped raw image
        imc = resize_volume(imc) #
 
        # wouldnt there be different scales avoid artificial brightening of myo by introducing bloodpools

        full_image = Image.fromarray(np.uint8(cm.gray(imc_full)*255)).convert('L') # why is L importnat? 
        
        min_value_cropped_ui = np.min(cropped_unormalized_image) # introduce further distrubitno of images to emphasize variance of LGE
        max_value_cropped_ui = np.max(cropped_unormalized_image)


        # when normalizing with uint values, the values wrap back around, especially witht eh uint datatype.
        # when the values get subtracted to minimum the values are literally wrap around with uint 8.
        # therefore we dont want to deal with mask image; we shoudl specifically select the region.
        im2 = std_img(np.array(im.crop(bbox)), min_value_cropped_ui, max_value_cropped_ui)  # cropped raw image with myo only 
        im2 = resize_volume(im2)


        
       
        sc2 = std_img(np.array(scar_im.crop(bbox)))  # cropped lge segmentation
        sc2 = resize_volume(sc2)

        
        imc_full = resize_volume(imc_full, ex=128)  # full image these 

        # somehow the imc_full values are extremelyl smsall.
        
        # Save processed images
        base_name = os.path.splitext(filename)[0]
        
        # Handle NaN values
        im2[np.isnan(im2)] = 0
        sc2[np.isnan(sc2)] = 0

        # here I know that interpolation creates the effects
        #if base_name == '9952_PSIR' and slice_no == 2:
          #  pdb.set`_trace()
        #    assert im2[120, 60] == imc[120,60], f"slice: {slice_no} values: {im2[120,60]} {imc[120,60]}"

        
        # Save numpy arrays to what? why are these names important? 
        np.save(os.path.join(saving_folder, f'{base_name}_raw_{slice_no}.npy'), im2)
        np.save(os.path.join(saving_folder, f'{base_name}_cine_{slice_no}.npy'), imc)
        np.save(os.path.join(saving_folder, f'{base_name}_cine_whole_{slice_no}.npy'), imc_full)
        np.save(os.path.join(saving_folder, f'{base_name}_lge_{slice_no}.npy'), sc2)

def collect_and_save_json(processed_dir, output_json):
    """Collect all processed .npy files and save as JSON"""
    raw_, lge_, cine_, cine_whole_ = [], [], [], []

    for file in tqdm(os.listdir(processed_dir)):
        print(file)
        if not file.endswith('.npy'):
            continue
            
        if 'raw_' in file:
            raw_.append(np.load(os.path.join(processed_dir, file)))
            
            # Get corresponding files
            base = file[:file.index('raw_')]
            idx = file[file.index('raw_')+4:-4]
            
            lge_f = f'{base}lge_{idx}.npy'
            lge_.append(np.load(os.path.join(processed_dir, lge_f)))
            
            cine_f = f'{base}cine_{idx}.npy'
            cine_.append(np.load(os.path.join(processed_dir, cine_f)))
            
            whole_f = f'{base}cine_whole_{idx}.npy'
            cine_whole_.append(np.load(os.path.join(processed_dir, whole_f)))
    
    # Convert to arrays
    raw_ = np.array(raw_)
    lge_ = np.array(lge_)
    cine_ = np.array(cine_)
    cine_whole_ = np.array(cine_whole_)
    
    # Create data dictionary
    datas = {
        'lge_whole': cine_whole_, # CINE whole = input 
        'lge_cropped': cine_, # 
        'masked_input': raw_,
        'lge_seg': lge_
    }
    
    # Custom JSON encoder for numpy arrays
    def default(obj):
        if type(obj).__module__ == np.__name__:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj.item()
        raise TypeError('Unknown type:', type(obj))
    
    # Save to JSON
    with open(output_json, "w") as outfile:
        json.dump(datas, outfile, default=default)
    
    print(f"Saved data shapes:")
    print(f"Raw: {raw_.shape}")
    print(f"LGE: {lge_.shape}")
    print(f"Cine: {cine_.shape}")
    print(f"Cine Whole: {cine_whole_.shape}")

def main():
    args = parse_args()
    input_dir = args.input_dir
    
    # Output directories
    processed_dir = args.output_dir
    os.makedirs(processed_dir, exist_ok=True)
    
    # Process each .mat file
    for filename in os.listdir(input_dir):
        if filename.endswith('.mat') and any(seq in filename for seq in ['PSIR', 'LGE', 'MAG']):
            file_path = os.path.join(input_dir, filename)
            process_mat_file(file_path, processed_dir, filename)
    

    if args.save_json != True: 
       return 

    # Collect all processed files and save as JSON
    collect_and_save_json(processed_dir, "./processed_data.json")

if __name__ == "__main__":
    main()