# Dataset Instructions for Medicinal Plant Classification using EfficientNet-B3

## Dataset Overview
The dataset comprises high-resolution images of medicinal plants categorized into five classes:
- **Neem**
- **Tulsi**
- **Aloe Vera**
- **Mint**
- **Eucalyptus**

These images are essential for training and testing the deep learning model. The dataset includes morphological variations and environmental conditions to improve model generalization.

## Dataset Source
The dataset can be downloaded from the following Kaggle link:
[Indian Medicinal Leaves Dataset](https://www.kaggle.com/datasets/aryashah2k/indian-medicinal-leaves-dataset)

## Folder Structure
Organize the dataset in the following folder structure:
```
Dataset/
├── Training/
│   ├── Neem/
│   ├── Tulsi/
│   ├── Aloe_Vera/
│   ├── Mint/
│   └── Eucalyptus/
├── Validation/
│   ├── Neem/
│   ├── Tulsi/
│   ├── Aloe_Vera/
│   ├── Mint/
│   └── Eucalyptus/
└── Testing/
    ├── Neem/
    ├── Tulsi/
    ├── Aloe_Vera/
    ├── Mint/
    └── Eucalyptus/
```

## Preprocessing Steps
1. **Image Resizing**:
   All images should be resized to **224x224 pixels** to match the input size of the EfficientNet-B3 model.

2. **Normalization**:
   Normalize the pixel values to the range [0, 1] by dividing each pixel value by 255.

3. **Data Augmentation**:
   To improve model generalization, apply the following transformations:
   - Rotation: ±20 degrees
   - Horizontal Flip
   - Zoom Range: 0.2
   - Width/Height Shift: 20%

4. **Noise Reduction**:
   Apply a Gaussian filter or median filter to reduce image noise.

5. **Dataset Splitting**:
   - **Training Set**: 80% of the dataset
   - **Validation Set**: 10% of the dataset
   - **Testing Set**: 10% of the dataset

## How to Prepare the Dataset
1. **Download the Dataset**:
   Use the Kaggle API or download the dataset manually from the link provided above.
   ```bash
   kaggle datasets download -d aryashah2k/indian-medicinal-leaves-dataset
   ```

2. **Unzip the Dataset**:
   Extract the downloaded dataset into a folder named `Dataset/`.
   ```bash
   unzip indian-medicinal-leaves-dataset.zip -d Dataset
   ```

3. **Organize the Folders**:
   Ensure the images are sorted into the `Training/`, `Validation/`, and `Testing/` directories as per the folder structure mentioned above.

4. **Apply Preprocessing**:
   Use a script to automate the resizing, normalization, and augmentation of images. Below is an example:
   ```python
   from tensorflow.keras.preprocessing.image import ImageDataGenerator

   datagen = ImageDataGenerator(
       rescale=1.0/255,
       rotation_range=20,
       width_shift_range=0.2,
       height_shift_range=0.2,
       shear_range=0.2,
       zoom_range=0.2,
       horizontal_flip=True,
       validation_split=0.1
   )
   ```

## Verifying the Dataset
After preprocessing, verify the dataset with the following steps:
1. Check if all images are resized to **224x224 pixels**.
2. Ensure that each class has an equal distribution of images in training, validation, and testing sets.
3. Verify the augmented images visually to ensure quality.

## Notes
- Ensure you have proper permissions to use the dataset in your project.
- If the dataset size is large, consider using cloud storage solutions like Google Drive or AWS S3 for accessibility.
- For issues or troubleshooting, refer to the Kaggle dataset discussion forum.

By following these instructions, you will prepare a high-quality dataset ready for training the deep learning model for medicinal plant classification.

