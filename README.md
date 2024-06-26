# Image Recognition for Shoe SKU Identification

This project aims to identify shoe models and specific SKUs using image recognition. It is built to incorporate into monitors and notify users of profitable SKUs. This can be particularly helpful for stores that obfuscate product titles but still upload correct images, such as TheBetterGeneration. Also, for the shoe silhouette model, using grayscale will probably be a better option, since the color of the shoe does not matter. It would make sure the model focuses on the shape of the shoe.

## Table of Contents

- **Overview**
- **Specifications**
- **Dataset Structure**

## Overview

- **Project Goal**: Use image recognition to identify shoe models and narrow down to specific SKUs.

- **Target Application**: Notify's monitors and profitable SKU pings.

- **Potential Use Case**: A TheBetterGeneration bot that only runs for profitable SKUs.

## Specifications

- **Training**: ~70 -> ~115 images per folder
- **Testing**: ~15 -> ~25 images per folder
- **Validation**: ~15 -> ~25 images per folder

## Dataset Structure

The dataset will be divided into three folders: Training, Testing, and Validation. Images will be distributed across these folders as follows:

- **Training Folder**: Contains 70 images per category for training the model.
- **Testing Folder**: Contains 15 images per category for testing the model's performance.
- **Validation Folder**: Contains 15 images per category for validating and fine-tuning the model.

## Notes

Using a pre-trained image recognition model is definitely the way to go as it allows for the model to learn faster and get higher accuracy.
I do not believe that VGG16.py uses any type of data augmentation, so it is possible that the model is simply memorizing the images, and not actually learning the shapes. If I were to add any type of data augmentation, it would use slight rotation and horizontal flipping. I might test with a VGG16_2.py that uses data augmentation and see if that performs better on tests.
