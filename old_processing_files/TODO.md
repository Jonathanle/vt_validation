# TODO:


I realize that this week, I should be combining all of my prevoius skills in model building as well as debugging in order to 
efficiently build a baseline preliminary model for scoring the likelihood of VA. Here is what I have done so far 


1. Created a dataset full of mat files + contours
2. Have a basic understanding of the outputs of the segmentation model as well as the dat


Now given all of this, I want to finally create a model that assesses the probability for a whole patient having VA. 

Some heuristic directions / df paths: 


## Get data running through first part of the network
1. Take a look at the dataset - determine its importance and how we can use this dataset for inference
- look at the inference for it
- run inference with the mat files. 
- take a quick look at the patientt slices with mat.

### Insights

- I learned that the preprocessing files should create a mask for the epi region --> I shoudl use this in the data to help the model learn that these are significant regions.'
- most of Nivetha's preprocessing code should handle matlalb processing --> test this out.


- this is here so that I have the computer having the representation to run the files - also can I have the representation for what I am looking at? 



## Build the head for the network to look at 1 slice see how well it does, then look for 3 slices