### Design
# Pre-training
* Include medical images of brain
* May include multi-modal images MRI, PET, CT, etc
* Unlabelled data, unpaired data

# Fine-tuning
* Model will use features used during pre-training 

# Model Architecture
* Segmented architecture
* Freezing different components should be straightforward
* Encoder shall be modular
* Segmentation shall be modular

# Code architecture
* Augmentations class
* SSLModel class
* Loss class (contrastive, rotational, recon, patch-wise reconstruction)
* Pre-training script
* Fine-tuning script

# Implemented encoders
* ViT

# Model Tracking
* mlflow tracking hooks