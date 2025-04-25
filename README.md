## Diabetic Retinopathy Classification Preprocessing Evaluation
### Get The Data
This project uses two datasets from kaggle:
1. [Aptos 2019 dataset](https://www.kaggle.com/competitions/aptos2019-blindness-detection)
2. [Diabetic Retinopathy Detection (EyePACS) dataset](https://www.kaggle.com/competitions/diabetic-retinopathy-detection)

Download both of these datasets, you only need the train folders. Set them up in the following file structure:

```
├── Diabetic_Retinopathy_Preprocessing
    ├── data
        ├── aptos2019
            ├── train_images
        ├── diabetic_retinopathy
            ├── train
```
### Explore Preprocessing Methods
We use five preprocessing methods in this project.
1. Regular: done before every other preprocessing method, includes resizing, random resized croping, normalizing, rotating, and horizontal flipping
2. CLAHE with Gaussian Filter: enhances local contrast using Contrast Limited Adaptive Histogram Equalization (CLAHE) while smoothing image noise and details with a Gaussian blur for better visual quality and feature extraction.
3. Histogram Equalization with Median Filter: globally enhances image contrast through histogram equalization while reducing salt-and-pepper noise using a median filter to preserve edges and fine details.
4. CLAHE on Green Channel with Median Filter: targeted image enhancement approach where local contrast is improved by applying CLAHE specifically to the green channel—often the most informative in medical and natural images—followed by a median filter to suppress noise while preserving edges.
5. Gaussian Subtractive Normalization: enhances local contrast by subtracting a Gaussian-blurred version of the image from the original, effectively normalizing illumination variations and emphasizing fine structural details.

To generate a visualization of each preprocessing method done on the same image, run the cells in the [preprocessing notebook](preprocessing.ipynb)

### Set Up Virtual Environment
In order to run the code you first need to create a virtual environment and install the dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

### Fine-tune ResNet-50
ResNet code is found in the [resNet notebook]()

1. 

### Fine-tune EfficientNet
EfficientNet code is found in the [efficientNet notebook](efficientNet.ipynb)

1. Cell 1 loads functions for loading the data, loading the model, training, and testing.
2. To run all the functions in the right order run cell 2 and pass in the preprocessing method you want to fine-tune the model with (clahe_gaussian_blur,  hist_equalization_median_blur, clahe_green_channel, or gaussian_subtractive_normalization).
3. The last cell has a test_from_checkpoint function that you can run to test the model from model weights saved during training.

### Fine-tune DenseSwin
DenseSwin code is found in the [denseNet notebook](denseNet.ipynb)

1. Cell 1 loads functions for loading the data, loading the model, training, and testing.
2. To run all the functions in the right order run cell 2 and pass in the preprocessing method you want to fine-tune the model with (clahe_gaussian_blur,  hist_equalization_median_blur, clahe_green_channel, or gaussian_subtractive_normalization).
3. The last cell has a test_from_checkpoint function that you can run to test the model from model weights saved during training.

### Visualize Metrics From All Models
To generate a multi-chart bar graph visualization of a performance comparison of ResNet-50, EfficientNet, and DenseSwin across all five preprocessing techniques run the cell in the [plot results notebook](plot_results.ipynb)

