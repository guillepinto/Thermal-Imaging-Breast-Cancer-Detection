**Notice:** This repository has been archived as the project has evolved into a more advanced version aimed at research publication. The current research, which includes fine-tuned models and improved results, is now available [here](https://github.com/guillepinto/Fine-Tuned-Thermal-Breast-Cancer-Diagnosis). The latest work will be presented at [U24 Fest](https://u24.uis.edu.co/), and aims for publication in peer-reviewed conference proceedings.
<div align="center">

# Thermal Imaging for Breast Cancer Detection


Guillermo Pinto, Miguel Pimiento, Oscar Torrens, Cristian Hernandez<br>

### [Research Report](reports/Breast_Cancer_Diagnosis_Based_on_CNNs_Using_Thermal_Imaging.pdf)

> *This project focuses on using thermal imaging as a non-invasive method for detecting breast cancer. Thermal imaging, also known as thermography, is a non-destructive, non-contact, and rapid technique that reports temperature by measuring the infrared radiation emitted by the surface of an object. This repository contains the research, implementation, and documentation of our work in this field.*

</div>

</br>

<p align="center">
<img src="assets/research_methodology.png">
</p>

**Overview of the Project:** Our approach involves using thermal images to detect anomalies indicative of breast cancer. The project consists of several stages: data collection, preprocessing, hyperparameter optimization, model training, and evaluation. We employ Convolutional Neural Networks (CNN) for the classification tasks.

## Dataset

Our dataset consists of thermal images, each infrared image has a dimension of 640 × 480 pixels; the soft­ware creates two types of files: (i) a heat-map file; (ii) a matrix with 640 × 480 points, e.g. 307,200 thermal points. Download the thermal image dataset from the [link provided](https://www.kaggle.com/datasets/asdeepak/thermal-images-for-breast-cancer-diagnosis-dmrir).

## Project Structure

- `notebooks/`: Jupyter notebooks with exploratory data analysis, experimental results and all the scripts for data preprocessing, feature extraction, and model training.
- `reports/`: Documentation and reports related to the project.

## Environments
To replicate the environment used in our experiments, please follow the instructions below.

### Setup
```console
# Clone the repository and navigate to the root directory:
git clone git@github.com:gpintoruiz/Thermal-Imaging-Breast-Cancer-Detection
cd Thermal-Imaging-Breast-Cancer-Detection

# Set up the Python virtual environment and activate it:
python3 -m venv venv
# For Unix-based systems (Linux/Mac)
source venv/bin/activate
# For Windows systems
# venv\Scripts\activate

# Install required libraries
pip install -r requirements.txt

# (Optional) Install and run jupyter notebook:
pip install notebook
jupyter notebook
```

## Acknowledgements

This project is based on the research methodology presented in the paper by Juan Pablo Zuluaga, Zeina Al Masry, Khaled Benaggoune, Safa Meraghni & Noureddine Zerhouni. [A CNN-based methodology for breast cancer diagnosis using thermal images](https://www.tandfonline.com/doi/full/10.1080/21681163.2020.1824685)

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/gpintoruiz/Thermal-Imaging-Breast-Cancer-Detection/blob/main/LICENSE) file for details.

## Contact

For inquiries, please contact: guillermo2210069@correo.uis.edu.co
