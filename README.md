# fetal-brain-ultrasound-deep-learning
-------------------------------------------------------------------------------------------------------------------------------
**Project Objective:**

The primary objective of this research project is to develop an automated system for the early detection and categorization of fetal brain abnormalities using ultrasound images. This system aims to enhance the visibility of abnormalities in ultrasound images, segment the fetal cranium and brain regions accurately, and classify the fetal brain conditions as either normal or anomalous. The ultimate goal is to provide healthcare personnel with a reliable and efficient tool for prenatal diagnosis, leading to improved patient outcomes.
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Dataset Description:**

The dataset used in this project consists of fetal ultrasound images. These images include both normal and anomalous cases of fetal brain conditions. The dataset is labeled and divided into two parts, with a 4:1 split for training and validation. Additionally, a set of 2949 unlabelled fetal ultrasound images is used for testing the developed model.
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Methodology:**

1. **Ultrasound Image Enhancement and Segmentation:** The project begins with the enhancement of ultrasound images to improve the visibility of abnormalities. Following this enhancement, a segmentation process is applied to isolate the cranium and brain regions in the images. Given the inherent noise in ultrasound pictures, a novel and innovative segmentation approach is implemented to achieve accurate results.

2. **Categorization into Brain Planes:** After segmentation, the enhanced and segmented images are further divided into three distinct brain planes, namely Trans-cerebellum, Transthalamic, and Trans-ventricular. This division is performed using a convolutional autoencoder.

3. **Training the Convolutional Autoencoder:** A convolutional autoencoder model is trained using the labeled dataset, which includes images categorized as either normal or anomalous. The model is designed to minimize the Mean Squared Error (MSE). Notably, the model exhibits remarkable performance with a low MSE value of 0.00297. The training process is monitored using loss curves, which indicate that the model effectively learns and generalizes after the 15th epoch.
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Results:**

The project yields impressive results, demonstrating the high efficiency and fit of the developed model. Specifically:

- The trained convolutional autoencoder model successfully categorizes fetal ultrasound images as either abnormal or normal.
- Testing the model on a set of 2949 unlabelled fetal ultrasound images leads to accurate classification.
- By automating the identification of irregularities in fetal brain conditions, this methodology can significantly assist healthcare personnel in making more reliable and quicker diagnoses. This automation has the potential to improve patient outcomes by enabling early detection and treatment of neurological abnormalities in fetuses.
