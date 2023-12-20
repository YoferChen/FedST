#!/bin/bash

# Dataset for the material microscopic image segmentation task
# Reference: private

# Dataset for the medical image segmentation
# Reference: https://chaos.grand-challenge.org/Combined_Healthy_Abdominal_Organ_Segmentation/

# Dataset for the face segmentation task with different races
# Reference: https://github.com/switchablenorms/CelebAMask-HQ
# List_of_category = ['background', 'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear',
#                      'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']

# Dataset for the face segmentation task with different ages
# Reference: https://github.com/JingchunCheng/All-Age-Faces-Dataset
# Labeled mask: Manual annotation by professionals
# List_of_category = ['background', 'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear',
#                      'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth', 'beard']

# Train UNet for the face segmentation task with different ages
# If use pretrained model, please add param like: --pretrained_model ./pretrained_model/Face_Segment/model43_folds0_best.pkl
# Options of dataset: 'material', 'medical', 'face',  'aaf_face'
python ./train.py --dataset aaf_face --model unet --federated_algorithm fedavg
python ./train.py --dataset aaf_face --model unet --federated_algorithm fedprox
python ./train.py --dataset aaf_face --model unet --federated_algorithm feddyn
python ./train.py --dataset aaf_face --model unet --federated_algorithm feddc
python ./train.py --dataset aaf_face --model unet --federated_algorithm fedst_separate
python ./train.py --dataset aaf_face --model unet --federated_algorithm fedst_join

# Train UNet of the face segmentation task with different ages
# $model_path is the path of the pretrained model, like: ./dataset/AAF_Face/model_fedprox_0808_172044/model49_folds
python ./test.py --dataset aaf_face --fold 0 --model_path $model_path --algorithm fedavg
python ./test.py --dataset aaf_face --fold 0 --model_path $model_path --algorithm fedprox
python ./test.py --dataset aaf_face --fold 0 --model_path $model_path --algorithm feddyn
python ./test.py --dataset aaf_face --fold 0 --model_path $model_path --algorithm feddc
python ./test.py --dataset aaf_face --fold 0 --model_path $model_path --algorithm fedst_separate
python ./test.py --dataset aaf_face --fold 0 --model_path $model_path --algorithm fedst_join

# Train FedST_Separate DDPM for the face segmentation task with different ages
# Please reference: https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models

# Train FedST_Join DDPM for the face segmentation task with different ages
# if use pretrained model, please add param like: --pretrained_model_for_DDPM ./pretrained_model/DDPM_for_AAF_Face/40_Network.pth --pretrained_model ./pretrained_model/Face_Segment/model43_folds0_best.pkl
python ./train.py --dataset aaf_face --model fedst_ddpm --federated_algorithm fedavg