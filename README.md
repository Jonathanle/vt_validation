# VT-scoring
A network that assesses the risk of a patient suffering from Ventricular Arrythmia. Uses patient LGE images to analyze scar data.




# Data Preprocessing Pipeline


Data from our LGE dataset was preprocessed using Nivetha's mat_preprocessing ipynb. This required a file that transformed the directory of (patientname)_PSIR
mat files to a directory where each mat file was nested in a folder of the patient's name


## Considerations of Data

One potential issue that I find with the dataset is the lack of variability or a standardized way of quantifying the slices


## Considerations of Performance / Good Explainability of Data
- What kind of performance should I be getting? how do i handle imbalanced datasets? 
- 




### 
AUC ideas - representing probabilyt of positive instance being classified over lower


- by increasing threshold --> we consider as we add more negative samples the proportion of positive isntances with respect to each tiny change in d. 