import pdb
"""
scoring_model.py

File for storing all the models classes for 

"""


"""
TODO: Learn all t he syntax and specific nuances to reduce the cognitive load when interpreting complex systems
Force retrieval of key importnat concepts for helping later verify and create the models

# force retrieval of important representation for absolute intuitive understanding and representation / understanding
of system .

A force retrievall is theoreitcally faster as I avoid the debt of debugging and complexity and hard cognitive load when building
I can only go fast if I have a true understanding even with the claude AI on what is happening


TODO: Devise a system for highlighting specific mistakes for claude to text process
"""

import torch

# what is the module nn?  nn is the placeholder name for torch.nn
import torch.nn as nn
from attention_model import AttentionUNet

# TODO: Create model that takes in an image and ouputs a score1


class BaselineScorer(nn.Module):
    def __init__(self, model_save_path = './models/unet_att_focal_dice350.pt', n_slices = 3, freeze_backbone = True): 
        # error: forgot to initialize upper module to get upper functions
        super().__init__()


        # Defining Parameters of LGE Segmenter

        # Define the model to make
        self.LGE_segmenter = AttentionUNet()

        # Error: How do I load the parameters: use load_state_dict on the model
        # Here I need to load the "state" dict object --> this thing has all the parameters
        # then call the state dict fucntion to transfer paremterss to model 
        model_parameters = torch.load(model_save_path) # warning
        self.LGE_segmenter.load_state_dict(model_parameters)

        # Error / Missing Objective - Freeze the Attention UNet Here? vs the optimizer? 
        if freeze_backbone: 
            for param in self.LGE_segmenter.parameters(): 
                param.requires_grad = False


        # Error: in order to compute AttentionUNet with lineara I have to flattten 2d --> 1d
        self.flatten = nn.Flatten()

        # How to Customize the dense layer? any relevant considerations? 
        # errors - NOT dense --> LLinear , additionally parameters: just need input and output (omit in and out)
        self.dense_layer = nn.Linear(128 * 128 * n_slices, 500)
        self.dense_layer2 = nn.Linear(500, 1)


        self.intermediate_segmented_images = None

        # df how could i verify the output dimensions of Attention U_net? 
    def forward(self, x):  

        # Emphasize this computation in forward to be able to debug outputs rather than using nn sequential.
        # for every image --> create a new image.
        batch_size, num_images, height, width = x.shape

        x = x.reshape(-1, 1,  height, width) #componse image
        
        # does this corretly take the batch size? 
        y = self.LGE_segmenter(x) # run all images foward with segmenter in batch it will act like we are doing a bunch more examples
        # TODO: evaluate if keeping the llogits or normalizing the would be apprpriaate?
        y = y.view(batch_size, num_images, height, width)


        self.intermediate_segmented_images = y.clone().detach()
        # after segmentation, we need to work on the specific batch

        y = self.flatten(y)
        y = self.dense_layer(y) # these models are maybe creating too much noise in the segmentation
        y = self.dense_layer2(y)
        

        y = y.squeeze(1)
        # just return y 

        return y 


class CNNScorer(nn.Module):
    def __init__(self, model_save_path = './models/unet_att_focal_dice350.pt', n_slices = 3, freeze_backbone = True): 
        # error: forgot to initialize upper module to get upper functions
        super().__init__()


        # Defining Parameters of LGE Segmenter

        # Define the model to make
        self.LGE_segmenter = AttentionUNet()

        # Error: How do I load the parameters: use load_state_dict on the model
        # Here I need to load the "state" dict object --> this thing has all the parameters
        # then call the state dict fucntion to transfer paremterss to model 
        model_parameters = torch.load(model_save_path) # warning
        self.LGE_segmenter.load_state_dict(model_parameters)

        # Error / Missing Objective - Freeze the Attention UNet Here? vs the optimizer? 
        if freeze_backbone: 
            for param in self.LGE_segmenter.parameters(): 
                param.requires_grad = False
        
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(in_channels=n_slices, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        # Calculate the size for the linear layer dynamically
        self.flatten = nn.Flatten(1)
        # Assuming input_height and input_width are provided
        self.linear = nn.Linear(524288 , 1)

        self.intermediate_segmented_images = None

        # df how could i verify the output dimensions of Attention U_net? 
    def forward(self, x):  

        # Emphasize this computation in forward to be able to debug outputs rather than using nn sequential.
        # for every image --> create a new image.
        batch_size, num_images, height, width = x.shape

        x = x.reshape(-1, 1,  height, width) #componse image
        
        # does this corretly take the batch size? 
        y = self.LGE_segmenter(x) # run all images foward with segmenter in batch it will act like we are doing a bunch more examples




        y = y.view(batch_size, num_images, height, width)
        y = self.sigmoid(y)



        self.intermediate_segmented_images = y.clone().detach()
        
        y = self.relu(self.bn1(self.conv1(y)))
        y = self.maxpool(y)
        y = self.relu(self.bn2(self.conv2(y)))

        y = self.flatten(y)
        y = self.linear(y)


        y = y.squeeze(1)
        # just return y 

        return y 
    def plot_segmentations(self, y = None):
        import matplotlib.pyplot as plt 
        fig, ax = plt.subplots(4, self.intermediate_segmented_images.shape[1])
        images_cpu = self.intermediate_segmented_images.cpu().numpy()
        for patient in range(4): 
            for i in range(self.intermediate_segmented_images.shape[1]):
                ax[patient][i].imshow(images_cpu[patient][i], cmap = 'gray')
        print(y) 

        plt.show()

class CNNScorerWithMasks(nn.Module):
    """
    CNN Scorer that processed LGE segmentations with data 
    """
    def __init__(self, model_save_path = './models/unet_att_focal_dice350.pt', n_slices = 3, freeze_backbone = True): 
        # error: forgot to initialize upper module to get upper functions
        super().__init__()


        # Defining Parameters of LGE Segmenter

        # Define the model to make
        self.LGE_segmenter = AttentionUNet()

        # Error: How do I load the parameters: use load_state_dict on the model
        # Here I need to load the "state" dict object --> this thing has all the parameters
        # then call the state dict fucntion to transfer paremterss to model 
        model_parameters = torch.load(model_save_path) # warning
        self.LGE_segmenter.load_state_dict(model_parameters)

        # Error / Missing Objective - Freeze the Attention UNet Here? vs the optimizer? 
        if freeze_backbone: 
            for param in self.LGE_segmenter.parameters(): 
                param.requires_grad = False
        
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(in_channels=96, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # Calculate the size for the linear layer dynamically
        self.flatten = nn.Flatten(1)
        # Assuming input_height and input_width are provided
        self.linear = nn.Linear(131072, 1)

        self.intermediate_segmented_images = None

        # df how could i verify the output dimensions of Attention U_net? 


        
    def forward(self, x):  

        # Emphasize this computation in forward to be able to debug outputs rather than using nn sequential.
        # for every image --> create a new image.
        batch_size, num_images, channel, height, width = x.shape

        x_lge_images_only = x[:, :,0, :, :]

        x_lge_images_only = x_lge_images_only.reshape(-1, 1,  height, width) #componse image
        
        # does this corretly take the batch size? 
        y = self.LGE_segmenter(x_lge_images_only) # run all images foward with segmenter in batch it will act like we are doing a bunch more examples

        y = y.view(batch_size, num_images, height, width)
        y = self.sigmoid(y)


        self.intermediate_segmented_images = y.clone().detach()


        x_mask_images_only = x[:, :, 1, :, :]

        segmentations_with_mask = torch.stack([y, x_mask_images_only], axis = 2)


        segmentations_with_mask = segmentations_with_mask.view(-1, 2, 128, 128) # how does this work? 

        # y refers to the  
        y = self.relu(self.bn1(self.conv1(segmentations_with_mask)))
        y = self.maxpool(y)

        # go back to the batch sizes

        y = y.view(batch_size, -1, 64, 64)



        y = self.relu(self.bn2(self.conv2(y)))

        y = self.flatten(y)
        y = self.linear(y)


        y = y.squeeze(1)
        # just return y 

        return y 
    def plot_segmentations(self, y = None):
        import matplotlib.pyplot as plt 
        fig, ax = plt.subplots(4, self.intermediate_segmented_images.shape[1])
        images_cpu = self.intermediate_segmented_images.cpu().numpy()
        for patient in range(4): 
            for i in range(self.intermediate_segmented_images.shape[1]):
                ax[patient][i].imshow(images_cpu[patient][i], cmap = 'gray')
        print(y) 

        plt.show()


class RiskAssessmentModel(nn.Module):
    def __init__(self, model_save_path = './models/unet_att_focal_dice350.pt', n_slices = 3, freeze_backbone = True):
        super().__init__()
        
        # Define the model to make
        self.LGE_segmenter = AttentionUNet()

        # Error: How do I load the parameters: use load_state_dict on the model
        # Here I need to load the "state" dict object --> this thing has all the parameters
        # then call the state dict fucntion to transfer paremterss to model 
        model_parameters = torch.load(model_save_path) # warning
        self.LGE_segmenter.load_state_dict(model_parameters)

 
        self.sigmoid = nn.Sigmoid()
        # Risk assessment head
        self.risk_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(9600, 50),

            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(50, 1)  # Continuous risk score
        )
    
    def forward(self, x):
        # TODO: codify and verify the testing functionaltiy fo the network  thorugh inspection

        batch_size, num_images, height, width = x.shape
        x = x.reshape(-1, 1,  height, width) # Run all the images in parallel - get all batches


        y = self.LGE_segmenter(x) # run all images foward with segmenter in batch it will act like we are doing a bunch more examples
        y = y.view(batch_size, num_images, height, width)
        y = self.sigmoid(y) # fucntion for running all logits through normalizer


        self.intermediate_segmented_images = y.clone().detach()


#        pdb.set_trace()

       # y = y.reshape(-1, 1,  height, width) # why did i do this? there is no need

        self.pool = nn.AdaptiveAvgPool2d(40)
        y = self.pool(y) 
        #y = y.view(batch_size, num_images, 40, 40)

        risk_score = self.risk_head(y)

        risk_score = risk_score.squeeze(1)
        return risk_score
     
    def plot_segmentations(self, patient_ids = None, y = None):
        import matplotlib.pyplot as plt 
        fig, ax = plt.subplots(4, self.intermediate_segmented_images.shape[1])
        images_cpu = self.intermediate_segmented_images.cpu().numpy()
        
        if patient_ids == None: 
            patient_ids = [0, 0, 0, 0]
        if y == None: 
            y = [0, 0, 0, 0]

        for idx, (patient, case) in enumerate(zip(patient_ids, y), 0): 
            for i in range(self.intermediate_segmented_images.shape[1]):
                ax[idx][i].imshow(images_cpu[idx][i], cmap = 'gray')
                ax[idx][i].set_title(f'patient: {patient} case: {case}')

        plt.show()

