#TODO: Determine why this processing of the file is important <---> what is it doing? 
#TODO: Install dependencies to run and then test --> we can document + reduce certaintyu on how to use
"""
Program such that given a directory, we transform to numpy files - is this correct

"""



# ##### Import all libraries

import scipy.io as sio
import scipy
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import tomni
from matplotlib import cm
from sklearn.model_selection import train_test_split
import torch


# ##### This snippet gets the list of .mat files under each subject

path= './data/Matlab/'
subjects= os.listdir(path)
for si in range(len(subjects)):
    new_path= path+subjects[si]
    files= os.listdir(new_path)
print('Total number of subjects: ', len(subjects))
print(files)


''' Sample: loading the last subject's PSIR file to check its keys '''
flge= sio.loadmat(new_path + '/'+ files[2])
flge.keys()

# #### Saving Cropped Images


''' Standardization and resizing functions '''

def std_img(tens):
    t_ = (tens-np.amin(tens))/(np.amax(tens)-np.amin(tens))
    return t_

def resize_volume(img, ex=64):              ### THIS IS CURRENTLY OPERATING ON 2D DATA, 'ex' IS THE EXPECTED SIZE
    current_depth = img.shape[0]
    current_width = img.shape[1]            

    depth_factor = ex / current_depth
    width_factor = ex / current_width

    factors = (depth_factor, width_factor)

    return scipy.ndimage.zoom(img, factors, order=1)


''' Initialize all variables to start saving '''

path= './data/Matlab/'
subjects= os.listdir(path)
si = 0
yes_save = 1



def saver(si):
    saving_folder = './cropped_myo_lge_testing/'                                                            ### RENAME BASED ON FOLDER TO BE SAVED IN
    os.makedirs(saving_folder, exist_ok=True)
    if yes_save:
        try:
            os.mkdir(saving_folder +subjects[si])
        except:
            print('error creating folder; folder already exists: ', os.path.exists(saving_folder +subjects[si]))

    new_path= path+subjects[si]
    files= os.listdir(new_path)

    for fs in files:
        # slicer= (75, 175, 45, 145)  ### if you want to manually crop the images
        if 'PSIR' in fs or 'LGE' in fs or 'MAG' in fs:
            flge= sio.loadmat(new_path+'/'+fs)
            try:
                __ = flge['series_type']
            except:
                print('no series for ', fs)
                continue
            assert flge['series_type']==np.array(['Myocardial Evaluation'])

            print(('Length: ', flge['enhancement'][0].shape[0]))

            for slice_no in range(flge['enhancement'][0].shape[0]):
                scar= np.copy(flge['enhancement'][0][slice_no]).astype('float')
                scar[scar==0]=np.nan
                if np.nansum(scar)!=0:
                # if 1:
                    try:
                        _= flge['lv_endo'][0][slice_no][0][0]
                    except:
                        print('couldnt get lv_endo')
                        continue
                    img_shape= np.transpose(flge['raw_image'][0,slice_no]).shape
                    myo_seg_endo= tomni.make_mask.make_mask_contour(img_shape, 
                                                                    flge['lv_endo'][0][slice_no][0][0][0])    ### CONVERT CONTOURS TO BINARY MASKS
                    myo_seg_epi= tomni.make_mask.make_mask_contour(img_shape,
                                                                flge['lv_epi'][0][slice_no][0][0][0])         ### CONVERT CONTOURS TO BINARY MASKS
                    myo_seg= (myo_seg_epi - myo_seg_endo).astype('float')
                    flge['raw_image'][0,slice_no]/=np.amax(flge['raw_image'][0,slice_no])
                    myo_seg[myo_seg==0]= np.nan
                    fin_img= flge['raw_image'][0,slice_no]*myo_seg
                    imc_ = flge['raw_image'][0,slice_no]
                    imc_full = std_img(np.array(flge['raw_image'][0,slice_no]))
                    fin_img[np.isnan(fin_img)]=0                                                              ### NEEDED TO SAVE


                    im= Image.fromarray(np.uint8(cm.gray(fin_img)*255)).convert('L')
                    imc__ = Image.fromarray(np.uint8(cm.gray(imc_)*255)).convert('L')
                    scar_im= Image.fromarray(np.uint8(cm.gray(scar)*255)).convert('L')

                    ''' Get bounding boxes from contours and crop the images. '''
                    im.getbbox()                                                                              
                    im2 = (std_img(np.array(im.crop(im.getbbox()))))   ## cropped raw image with myo only
                    im2 = resize_volume(im2)

                    imc = (std_img(np.array(imc__.crop(im.getbbox()))))  ## cropped raw image
                    imc = resize_volume(imc)

                    sc2 = (std_img(np.array(scar_im.crop(im.getbbox()))))   ## cropped lge segmentation
                    sc2 = resize_volume(sc2)

                    ''' Use this to visualize the results being stored '''
                    # sc2[sc2==0]=np.nan
                    # im2[im2==0]=np.nan
                    # im2[im2==0]=np.nan
                    # plt.imshow(imc, cmap='gray')
                    # plt.plot(flge['lv_endo'][0][slice_no][0][0][0][:,0], flge['lv_endo'][0][slice_no][0][0][0][:,1])
                    # plt.plot(flge['lv_epi'][0][slice_no][0][0][0][:,0], flge['lv_epi'][0][slice_no][0][0][0][:,1])
                    # plt.imshow(myo_seg, cmap='gray')
                    # plt.colorbar()
                    # plt.imshow(sc2, cmap='jet')
                    # plt.imshow(imc_full, cmap='gray')
                    # plt.show()
                    # # plt.colorbar()
                    # plt.axis('off')

                    if yes_save:
                        
                        save_path= saving_folder +subjects[si] + '/raw_'+ str(slice_no) + '.npy'
                        im2[np.isnan(im2)]=0
                        np.save(save_path, im2)

                        save_path= saving_folder +subjects[si] + '/cine_'+ str(slice_no) + '.npy'
                        np.save(save_path, imc)

                        save_path= saving_folder +subjects[si] + '/cine_whole_'+ str(slice_no) + '.npy'
                        np.save(save_path, imc_full)

                        save_path=saving_folder +subjects[si] + '/lge_'+ str(slice_no) + '.npy'
                        sc2[np.isnan(sc2)]=0
                        np.save(save_path, sc2)



for si in range(len(subjects)):
    saver(si)

# ##### Intermediate .npy files used for train-test split, not for directly storing as JSON


key_dir = './cropped_myo_lge_testing/'

raw_=[]
lge_=[]
cine_=[]
cine_whole_=[]

for subject in os.listdir(key_dir):    
    files= os.listdir(os.path.join(key_dir, subject))
    for f in files:
        idx= f[4:list(f).index('.')]
        if 'raw' in f:
            raw_.append(np.load(os.path.join(key_dir, subject, f)))

            lge_f = 'lge_'+str(idx)+'.npy'
            lge_.append(np.load(os.path.join(key_dir, subject, lge_f)))

            lge_f = 'cine_'+str(idx)+'.npy'
            cine_.append(np.load(os.path.join(key_dir, subject, lge_f)))

            lge_f = 'cine_whole_'+str(idx)+'.npy'
            cine_whole_.append(resize_volume(np.load(os.path.join(key_dir, subject, lge_f)), ex=224))

raw_ = np.array(raw_)
lge_= np.array(lge_)
cine_= np.array(cine_)
cine_whole_ = np.array(cine_whole_)

raw_.shape, lge_.shape, cine_.shape, cine_whole_.shape

# ##### Store as JSON


import json

datas = {'lge_whole': cine_whole_,
         'lge_cropped': cine_,
         'masked_input': raw_, 
         'lge_seg': lge_}

def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))

with open("./sample_inference_data.json", "w") as outfile: 
    json.dump(datas, outfile, default=default)


# TODO: Evaluate if this is a reasonaable objective
inf_data = json.load(open("./sample_inference_data.json"))
inf_data.keys()


lge_whole= np.array(inf_data['lge_whole']).reshape(44,224,224)
x = np.array(inf_data['masked_input']).reshape(44,64,64)
y = np.array(inf_data['lge_seg']).reshape(44,64,64)
cropped = np.array(inf_data['lge_cropped']).reshape(44,64,64)


