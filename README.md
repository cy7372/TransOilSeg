# TransOilSeg: Oil Spill Detection with Deep Learning

**TransOilSeg** is a deep learning-based segmentation model designed for oil spill detection. 

## Installation and Setup

### Step 1: Install Dependencies
To begin, install the required dependencies using `pip`:

```shell
pip install -r requirements.txt
```
This will install all necessary libraries, including PyTorch, OpenCV, and other required packages.

### Step 2: Prepare the Dataset
To use the TransOilSeg model, you need to convert the dataset into the `.npz` format. This can be done using the provided script `datasets/img2npz.py`. 

Make sure your dataset follows the structure of `images` and `labels` directories, and the labels are in PNG format. The script will generate `.npz` files that can be used for both training and inference.

### Step 3: Run Inference
Once the dataset is prepared, you can run inference on your data. To do this, use the `inference.py` script. This script will load the pre-trained model and perform inference on your dataset.

Run the following command:

```shell
python inference.py
```

This will output the model's performance and save the results (segmentation masks) to the specified directory.

### Step 4: Continue Training (Optional)
If you want to continue training from a pre-trained model, you can use the `train.py` script. This script allows you to resume training with the pre-trained weights, and fine-tune the model on your specific dataset.

Run the following command to start or resume training:

```
python train.py
```

This will train the model using the dataset you've converted into `.npz` format. The model will be saved periodically during training.

## Model Training

### Customizing Training
To customize the training configuration, modify the `configs/M4D_shared.ini` file. This file contains parameters such as:
- **batch_size**: Size of the training batches
- **max_epochs**: Maximum number of epochs for training
- **img_size**: Image size for training (224x224 by default)
- **name_classes**: List of class names for segmentation


## Notes

- Ensure your dataset is in the correct format, with images and labels in the appropriate directories.