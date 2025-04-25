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
DenseSwin code is found in the [denseNet notebook](train_test_eval_rylee.ipynb)

1. Cell 1 loads functions for loading the data, loading the model, training, and testing.
2. To run all the functions in the right order run cell 2 and pass in the preprocessing method you want to fine-tune the model with (clahe_gaussian_blur,  hist_equalization_median_blur, clahe_green_channel, or gaussian_subtractive_normalization).
3. The last cell has a test_from_checkpoint function that you can run to test the model from model weights saved during training.

### Visualize Metrics From All Models
To generate a multi-chart bar graph visualization of a performance comparison of ResNet-50, EfficientNet, and DenseSwin across all five preprocessing techniques run the cell in the [plot results notebook](plot_results.ipynb)

