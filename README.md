# Extended Excitation Backprop with Gradient Weighting: A General Visualization Solution for Understanding Heterogeneous Face Recognition

This is the official implementation of our paper published in Pattern Recognition Letters:

Wang, Y., Kannappan, S., Bai, F., Gibson, S. and Solomon, C. (2025) "Extended Excitation Backprop with Gradient Weighting: A General Visualization Solution for Understanding Heterogeneous Face Recognition", Pattern Recognition Letters, Vol. 192, Pages 136-143

In this paper, we proposed a gradient-weighted extended Excitation Back-Propagation (gweEBP) method that integrates the gradient information during its backpropagation for the accurate investigation of embedding networks. We performed an extensive evaluation of our gweEBP, and seven other visualization methods, on two neural networks, trained for heterogeneous face recognition. The evaluation is performed over two publicly available cross-modality datasets using two evaluation methods termed the "hiding game" and the "inpainting game". 

# Installation
Tested with Python 3.7, PyTorch 1.9.0

# Networks
We provide two convolutional neural networks fine-tuned for HFR: VGGFace16 and LightCNN-9. The fine-tuned models can be downloaded from [here](https://drive.google.com/file/d/1MvpQtpMRurew60aHzWqUhPRtDzVqpvUG/view?usp=sharing) and should be unzipped as the ./models folder and placed under the root directory of this project.

# Data
The test data for evaluating visualization/attribution methods can be downloaded [here](https://drive.google.com/file/d/1Gewjuwsn5n3Kt9rLNFpJ_KMC0AXNF2YF/view?usp=sharing). Create a sub-folder named "data" and put the unzipped folder "inpainting-game" into this sub-folder.

# Run Inpainting Game

Firstly, change the work directory to "eval".

## Generating the saliency maps for the Inpainting Game

To generate saliency maps using whitebox attribution methods:

```python
python generate_inpaintinggame_wb_saliency_maps_multigpu_{dataset}.py --net {net} --method {white_box_method}
```

The dataset can be either 'UoM' or 'CUHK'. The net can be either 'vgg16' or 'lcnn9'. The white_box_method is one of 'EBP', 'cEBP', 'tcEBP', 'gweEBP' and 'gradcam'.

To generate saliency maps using blackbox attribution methods:
```python
python generate_inpaintinggame_bb_saliency_maps_multigpu_{dataset}.py --net {net} --method {black_box_method}
```
The dataset can be either 'UoM' or 'CUHK'. The net can be either 'vgg16' or 'lcnn9'. The black_box_method is one of 'CorrRISE', 'XFace' and 'PairSIM'.

## Evaluating and plotting the Inpainting Game

Once the saliency maps have been generated for all the attribution methods you want to compare, the inpainting game can be run to compare the performance of these methods as follows:

```python
python run_inpainting_game_eval_{dataset}.py --net {dataset}-{net} --output output/inpainting_game/{dataset}-{net} --method {methods_to_compare}
```

The dataset can be either 'UoM' or 'CUHK'. The net can be either 'vgg16' or 'lcnn9'. The methods_to_compare can be one or more of 'EBP', 'cEBP', 'tcEBP', 'gewEBP', 'GradCAM', 'bbox-xface', 'bbox-corrrise_perct=10_scale_12' and 'PairwiseSIM'. The default is to compare all the attribution methods. 

# Acknowledgement

Our implementation is based on the source code from the project "Explanable Face Recognition" (XFR) by J. Williford et al.: J. Williford, B. May and J. Byrne, "Explainable Face Recognition", ECCV 2020. We thank them for the sharing of their excellent work.
