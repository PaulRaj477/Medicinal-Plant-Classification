# README for Medicinal Plant Classification using EfficientNet-B3

## Project Overview
This project focuses on medicinal plant classification using a robust deep learning framework based on EfficientNet-B3 architecture. It is designed for accurate and efficient categorization of medicinal plants from image datasets, overcoming challenges like dataset heterogeneity and fine-grained morphological variations.

## Features
- Utilizes EfficientNet-B3 for feature extraction.
- Implements domain-specific transfer learning.
- Includes high-level preprocessing like data augmentation and noise reduction.
- Achieves classification accuracy between 95.8% and 98.9%.
- Optimized for real-time applications with computational scalability.

## Project Structure
```
|-- Source Code
|-- README.md
|-- requirements.txt
|-- Model (efficientnet_b3_medicinal_plant_classifier.h5)
|-- Dataset Instructions
|-- Documentation
   |-- Research Paper (Palraj Karuppasamy.docx)
   |-- Results and Analysis
|-- License
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/medicinal-plant-classification.git
   ```
2. Navigate to the project directory:
   ```bash
   cd medicinal-plant-classification
   ```
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Ensure the dataset is prepared and stored as specified in the Dataset Instructions.
2. Run the training script to train the model:
   ```bash
   python train_model.py
   ```
3. Use the saved model to perform predictions on new images:
   ```bash
   python predict.py --image_path /path/to/image.jpg
   ```

## Dataset
The dataset includes high-resolution images of medicinal plants categorized into five classes: Neem, Tulsi, Aloe Vera, Mint, and Eucalyptus. Refer to the Dataset Instructions for preprocessing and augmentation details.

## Results
The proposed model achieves high performance metrics:
- Accuracy: 95.8% - 98.9%
- Precision, Recall, F1-Score: > 94%

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Contributors
- **Palraj Karuppasamy**
- **V. Joseph Michael Jerard**
- **Rohini Ganapathi**

## Contact
For any queries or contributions, please contact:
- **Palraj Karuppasamy**: [palrajk@gmail.com](mailto:palrajk@gmail.com)
- **V. Joseph Michael Jerard**: [jerard.vedam@gmail.com](mailto:jerard.vedam@gmail.com)
- **Rohini Ganapathi**: [rohini.manoharan@gmail.com](mailto:rohini.manoharan@gmail.com)



