# Bird Classification Model

This bird classification model was trained using the NABirds Dataset from Cornell Lab of Ornithology.

https://dl.allaboutbirds.org/nabirds

* nabirds.py - Convert the dataset to something that can be consumed by Torch. This creates a compatibile folder structure, merges male/female/juvenile folders, and crops the targets for use in a Resnet.
* train.py - Simple resnet50 trainer. Supports resume.
* infer.py - Infer a single image given a checkpoint.
* export.py - Export a model to OpenVINO, CoreML, ONNX, and NCNN.

These models are for non-commercial use only, per the NABirds Dataset Terms of Use.

## Notes

This repository originally used the bird species dataset found here: https://huggingface.co/datasets/chriamue/bird-species-dataset

Though the dataset was larger, it was missing common species. Furthermore, the training images were possibly too "perfect" to provide good training data for identification at home feeders: the foreground targets were always in focus, and the background was blurred. This possibly made it difficult to train a model that was able to hande noise in the image.
The images were also pre-cropped to 224x224 making transformations extra lossy.
