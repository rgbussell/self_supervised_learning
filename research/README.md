# Pre-training comparisons
OpenMind evaluted pre-traing with 114k MRI cases with
particular focus on Primus-M (transformer) and ResEnc-L CNN architectures
* Reconstruction-based pre-training approaches outperform (MAE)
* Best segmentation performance: ResEnc-L CNN
* Primus-M underperformed ResEnc-L CNN
* Pre-training lead to faster convergence in fine-tuning phase

# From FDG to PSMA, Oct 2024 arxiv:2409.09478v2
* ResEncL nnU-Net architecture
* multi-task learning
* pre-training
* reduction in false positives and false negatives
* misalignment data augmentation
* multi-modal pre-training on CT, MR and PET data

# Pre-trained weights
* AutoPET III, ResEncUnetL Rokuss arxiv 2409:09478v2 (https://zenodo.org/records/13753413)
* 

# Large pre-training datasets
* OpenMind
* TotalSegmentator