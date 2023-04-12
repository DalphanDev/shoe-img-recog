# Image Recognition for Shoe SKU Identification

This project aims to identify shoe models and specific SKUs using image recognition. It is built for Notify to incorporate into their monitors and notify users of profitable SKUs. This can be particularly helpful for stores that obfuscate product titles but still upload correct images, such as TheBetterGeneration. Also, for the shoe silhouette model, using grayscale will probably be a better option, since the color of the shoe does not matter. It would make sure the model focuses on the shape of the shoe.

## Table of Contents

- **Overview**
- **Specifications**
- **Dataset Structure**

## Overview

- **Project Goal**: Use image recognition to identify shoe models and narrow down to specific SKUs.

- **Target Application**: Notify's monitors and profitable SKU pings.

- **Potential Use Case**: A TheBetterGeneration bot that only runs for profitable SKUs.

## Specifications

- **Training**: ~70 images per folder
- **Testing**: ~15 images per folder
- **Validation**: ~15 images per folder

## Dataset Structure

The dataset will be divided into three folders: Training, Testing, and Validation. Images will be distributed across these folders as follows:

- **Training Folder**: Contains 70 images per category for training the model.
- **Testing Folder**: Contains 15 images per category for testing the model's performance.
- **Validation Folder**: Contains 15 images per category for validating and fine-tuning the model.
