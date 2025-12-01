### Design
# Pre-training
* Include medical images of brain
* May include multi-modal images MRI, PET, CT, etc
* Pre-training should be performed with no or only easily obtainable labels

# Fine-tuning
* Model 

# Model Architecture
* Segmented architecture
* Freezing different components should be straightforward
* Encoder shall be modular
* Segmentation shall be modular

# Code architecture
* Augmentations class
* SSLModel class
* Loss class (contrastive, rotational, recon)
* Pre-training script
* Fine-tuning script

# Implemented encoders
* SimCLR
* SwinUNETR
* SwinUNETRv2

# Model Tracking
* add mlflow tracking hooks