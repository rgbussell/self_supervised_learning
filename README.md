## Objective: 
Develop an image-to-image translation method for PET to T1
(positron emission tomography to T1 magnetic resonance iamges)

## Constraints
T1 should be of sufficient image quality for accurate gray/white/ventricle segmentation using standard procesing protocols.<br>
Use only open source, publicly available data sources<br>
Trainable in < 48 hours single mid-grade GPU (specs below)<br>
Inference including preprocessing pipeline < 60 seconds<br>

![alt text](assets/image-3.png)

## Project plan
Source data from public challenge data sets ISLES, BRaTS, etc.<br>
Set up ETL pipeline using pipeline orchestrator (dagster)<br>
Use self-supervision paradigm to learn a multi-modal latent space representation<br>
Minimize dependence on difficult to train architectures<br>

## Planned Architecture
Three networks are planned

1. Vision transformer as encoder network
Stores latent multi-model representation
Patch-wise reconstruction loss with masked auto-encoder setup

2. Decoder for ViT in the pre-training stage (patch-wise reconstruction)

3. SwinUNET fine-tuned on T1|PET pairs for the primary image-to-image translation task

## Progress
Pre-trained transformer results

Example validation data set from the pre-training (<24 hours training)
![alt text](assets/image-1.png)

Example Fine-tuned on T1 to FLAIR data task (using pretrained ViT encoder)
Training time < 12 hours, 16 GB 
left-to-right: T1 image (input), FLAIR ground truth, network prediction<br>
![alt text](assets/image-2.png)

## PET image sourcing
Openneuro Monash data set available with PET and MRI data in same sessions
Preprocessing pipeline required to set up fine-tuning

Here is one slice from the Monash resting state fMRI-PET data set from open neuro.
With some preprocessing and registration to the corresponding MRI this could
serve as a fine-tuning data set for the PET->T1 task

![alt text](assets/image.png)

## Image preparation fmr PET-T1 image-to-image translation
cd ./self_supervised_learning/etl
python run_etl_pipeline.py

## PET-T1 image-to-image translation initial results
* 20 paired slices selected from one of the PET timepoints
* 150 epochs of fine-tuning
* pre-training on the mae task with multi-modal MRI (no PET in training)
* nuumber of subjects: Train: 9, Val: 2, Test: 2 

Result of PET to T1 image-to-image translation after 150 epochs
![alt text](assets/image_pet_t1_i2i_initial_finetuning_150.png)


## Additional Data for Fine Tuning PET-T1
Increase the number of paired slices in finetuning
Distribution:
Train: 9 subjects (360 slices)
Val: 2 subjects (80 slices)
Test: 2 subjects (80 slices)

![alt text](assets/pet_t1_i2i_moreslices.png)


## Citation
we thank these sources for code and ideas
Tao 2023: arXiv:2212.01108v3
Liu 2023: arXiv:2311.04049v1