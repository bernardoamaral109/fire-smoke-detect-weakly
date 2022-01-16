# Weakly Supervised Fire and Smoke Segmentation in Forest Images with CAM and CRF

## Abstract 

The number of publicly available datasets with annotated fire and smoke regions in wildfire scenarios is very scarce. To develop flexible models that can help firefighters protect the forest and nearby populations we propose a method for segmenting fire and smoke regions in images using only image-level annotations, i.e. simple labels that just indicate the presence or absence of fire and smoke . The method uses Class Activation Mapping (CAM) on multi-label classifiers of fire and smoke, followed by Conditional Random Fields (CRF) to accurately detect fire/smoke masks at the pixel-level. Due to the high correlation of fire and smoke labels, we found that a single classifier is unable to provide simultaneously good segmentation for fire and smoke. We propose the training of two classifiers of different complexities, one to support the segmentation of fire and the other for smoke. Compared with fully-supervised methods, the proposed weakly-supervised method is quite competitive, while requiring much less labeling effort in dataset preparation.

## Proposal 

![alt text](readme/proposal.png?raw=true)


## Labelling Method
![alt text](readme/mask.png?raw=true)

## CAM

###### CAM approach

![alt text](readme/vgg19_cam_fire.png)

###### CAM heatmaps
![alt text](readme/fire_cam.png?raw=true)
![alt text](readme/smoke_cam.png?raw=true)

## CRF

![alt text](readme/crf.png?raw=true)

## Results

###### Fire results
![alt text](readme/compare_fire.jpg?raw=true)

###### Smoke results
![alt text](readme/compare_smoke.jpg?raw=true)


## Models 

 [Models used.](https://drive.google.com/drive/folders/1RDAnKYQEwi4H3YsE_c_SsSeYvbFQfuDl?usp=sharing).


