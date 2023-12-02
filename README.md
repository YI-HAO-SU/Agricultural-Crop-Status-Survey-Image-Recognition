# Agricultural-Crop-Status-Survey-Image-Recognition

The agricultural crop status survey operation involves personnel capturing images on-site with cameras and documenting the details. 
The extensive and fragmented nature of agricultural regions in our country, with numerous plots, results in a vast amount of image data. 
Converting this data into information usable for management system operations requires significant human effort and time. 
As AI technology continues to advance, image interpretation tasks have matured in recent years, making them highly suitable for integration into the agricultural crop status survey workflow. 
This integration can expedite the acquisition of necessary information for agricultural authorities.

While there are comprehensive AI datasets in areas such as daily life, industry, and healthcare, the agricultural domain is relatively lacking. 
Meeting the future demand for AI technology in smart agriculture will require substantial investment in professional manpower for the collection and analysis of agricultural-related information.
Therefore, the challenges presented in this competition aim to assist students in understanding the application needs of agricultural datasets and image recognition in the agricultural industry. 
It also aims to cultivate students' experience and technical capabilities in using AI technology for image recognition in the agricultural sector.

Weight: https://drive.google.com/drive/folders/1Hn5xQ2EL_i7t-jxMG-351PZHNk3-XmV1?usp=sharing

- [Data Observation and Analysis](#1)
- [Data Pre-processing](#2)
- [Model structure](#3)
- [Model training](#4)
- [Experimental Results](#5)



<h2 id="1">Data Observation and Analysis</h2>
The content of the dataset comprises on-site crop survey images, including poorly captured images such as houses, vehicles, agricultural machinery, and blurry scenes, totaling over 100,000 images across 33 categories.
Each data entry includes provided centroid coordinates.

![圖片](https://github.com/YeeHaoSu/Agricultural-Crop-Status-Survey-Image-Recognition/assets/90921571/32c56f35-7369-421c-9abf-107498fcc7c8) ![圖片](https://github.com/YeeHaoSu/Agricultural-Crop-Status-Survey-Image-Recognition/assets/90921571/2419fee5-9563-4057-a268-c6ce8dde2c16)

Due to significant variations in the sizes of images within the training dataset, ranging from 640x936 to 5760x4320, a preprocessing step was undertaken to standardize the resolutions. 
This standardized resolution was then used as a reference for subsequent data augmentation, as illustrated in the diagram below:

* The red box indicates the index of each category.

* The yellow box displays the name of the respective category.

* The green box represents the total number of training images in that category.

* The blue box indicates the resolution (a x b) of the images.

* The purple box specifies the number of images in that category belonging to the given resolution

![圖片](https://github.com/YeeHaoSu/Agricultural-Crop-Status-Survey-Image-Recognition/assets/90921571/975522d9-ee53-4689-b323-a543a81dd037)

Upon analyzing statistical charts, it became evident that the disparate sizes of the data could potentially pose challenges during training. 
To address this issue, various methods for resizing data were experimented with to alleviate training difficulties. 
Additionally, we observed distinct plant occurrences based on different latitudes. 
Therefore, we hypothesize that incorporating latitude conditions could enhance model training.

To test the effectiveness of latitude conditions on model performance, experiments were conducted to assess whether there was an improvement in accuracy. 
The experimental results will be analyzed to determine the impact of latitude conditions on the model's predictive capabilities. 
Through systematic experimentation based on the provided dataset conditions, we aim to identify which conditions contribute positively to the model's training and which do not.

**[⬆ back to top](#Agricultural-Crop-Status-Survey-Image-Recognition)**

<h2 id="2">Data Pre-processing</h2>
Due to significant variations in the sizes of images within the training dataset, an essential preprocessing step was implemented to enhance training efficiency. 
The approach involved resizing all images in the training dataset to a unified dimension of 456x456. 
Subsequently, a center crop operation was applied to further standardize the dataset.

Initially, it was planned to utilize the centroid provided in the dataset as a reference for center cropping.
However, upon observation, it was noted that the connection between the centroid position and the object was not consistently accurate. 
Consequently, the decision was made to disregard centroid-based data and instead use the center of the data as the reference for the center crop operation.

Additionally, data augmentation techniques, such as Snapmix, were employed to further enrich the dataset and enhance the model's robustness during training.
These strategies collectively aim to mitigate challenges arising from the disparate sizes of images in the training dataset and improve the overall training effectiveness.

![圖片](https://github.com/YeeHaoSu/Agricultural-Crop-Status-Survey-Image-Recognition/assets/90921571/a0002d20-db03-4426-81a8-2cd4cce5ba4a)

- Snapmix

1. Background:
    In the past, data augmentation methods like MixUp and CutMix primarily operated at the pixel level.
    For instance, CutMix utilizes the cutout area as a weighted mixture, but its effectiveness diminishes when the excised crucial parts have a small area.

2. Method:
    Snapmix is a data augmentation technique similar to CutMix but with a distinctive approach.
    Instead of calculating weights at the pixel level, Snapmix utilizes the Class Activation Map (CAM).
    The process involves two steps: label fusion and asymmetrically blended images.

    - Label Fusion: Initially, the CAM of an image is calculated, and a Semantic Percent Map (SPM) is obtained through normalization, with a total ratio of 1.
      The proportions of various classes are then calculated for label fusion.

    - Asymmetric Blended Images: For the creation of asymmetrically blended images, multiple images are cut at varying areas of unequal sizes and different positions.
      The blending is achieved through spatial transformations, introducing diversity to the augmented data.

3. Results:
    Snapmix has demonstrated promising results across multiple datasets, showing an average accuracy improvement of 12%. This indicates its efficacy in enhancing the performance of models trained on diverse datasets.

- Used Augmentation methods:

  - ColorJitter:
        Parameters: Contrast and saturation are both set to 0.5.
        This involves randomly adjusting the original values within the range (0.5, 1.5) to enhance data diversity.

  - Normalize:
        Parameters: Utilizes the mean and standard deviation from ImageNet for normalization.

  - Snapmix:
        Implementation using the timm library to introduce the Snapmix augmentation technique.


**[⬆ back to top](#Agricultural-Crop-Status-Survey-Image-Recognition)**


<h2 id="3">Model structure</h2>

Model: EfficientNet (pre-trained weight: b5)

- Preprocessing:

   - Resize: All images are resized to dimensions 456x456.

- Training Configuration:

  - Learning Rate: ExponentialLR with a decay factor (gamma) of 0.9.

  - Epochs: Training is conducted for 30 epochs.

![圖片](https://github.com/YeeHaoSu/Agricultural-Crop-Status-Survey-Image-Recognition/assets/90921571/167faf48-9ee2-4418-93a5-b85594bc1463)

**[⬆ back to top](#Agricultural-Crop-Status-Survey-Image-Recognition)**


<h2 id="4">Model training</h2>
From the knowledge of life, we can understand that the cultivation of fruits and vegetables is geographically specific, so we experimented with the correlation between geographic location and latitude and longitude with the types of crops grown. 
Therefore, we experimented with the correlation between geographic location latitude and longitude, and the types of crops grown, and used the following methods to experiment.

|                     | Latitude and longitude Accuracy | geographical location Accuracy |
| ------------------- | ------------ | ------------- |
| **Random Forest**   | 0.4171       | 0.4168        |
| Decision Tree       | 0.4169       | 0.4129        |
| SVM                 | 0.0312       | 0.0356        |
| KNN                 | 0.3523       | 0.3612        |
| Naive Bayes         | 0.0745       | 0.0677        |
| Logistic Regression | 0.1181       | 0.1095        |

Based on the experimental results, it was found that the Random Forest model yielded the best performance. 
Therefore, this method was adopted for further use. Interestingly, the experimental results related to geographical location closely resembled those of latitude and longitude. 
It is inferred that the two attributes are inherently close, as geographical location can be considered, to some extent, a representation of the distribution of latitude and longitude.
Due to the convenience of handling numeric data, subsequent experiments focused solely on latitude and longitude.

Additionally, we conducted experiments on the Top N Error.
Even if the accurate identification of the planted crops in the preferred geographical region was not achieved, it was expected that the correct answer would still be among the top N likely crops.
Therefore, the following experiments were conducted:

| TOP N Error | Accuracy               |
| ----------- | ---------------------- |
| TOP 1       | 0.4170917892385031     |
| TOP 5       | 0.7881586296778998     |
| **TOP 10**  | **0.9143176317259356** |
| TOP 16      | 0.9696518339229194     |

After obtaining the experimental data mentioned above, to avoid excessively influencing prediction results, we decided to provide a bonus to the top 10 predicted categories.
This action will take place after the classifier's output undergoes softmax processing across various probability regions for each class.

|           | Origin | +0.01  | **+0.1**    | +0.3    | +0.5   |
| --------- | ------ | ------ | ----------- | ------- | ------ |
| Accuracy  | 0.8260 | 0.829  | **0.8361**  | 0.8331  | 0.8329 |
| Variation | 0      | +0.003 | **+0.0101** | +0.0071 | 0.0069 |

In the ultimate decision-making process, an additional criterion was introduced to enhance the model's performance. A bonus of +0.1 points was assigned for correctly identifying the crop among the Top 10 categories. This served as a refinement to the accuracy metric, providing more weight to predictions falling within the Top 10.

In addition to calculating accuracy, the evaluation methodology was extended to include the computation of the F1 score. The F1 score is a metric that considers both precision and recall, providing a more comprehensive assessment of model performance, particularly in scenarios with imbalanced class distribution.

**[⬆ back to top](#Agricultural-Crop-Status-Survey-Image-Recognition)**


<h2 id="5">Experimental Results</h2>

![圖片](https://github.com/YeeHaoSu/Agricultural-Crop-Status-Survey-Image-Recognition/assets/90921571/a869b262-4a18-4345-a282-804300cc17b9) 
![圖片](https://github.com/YeeHaoSu/Agricultural-Crop-Status-Survey-Image-Recognition/assets/90921571/1e05cc95-9e21-45e9-9418-5ee1b65d914b)
![圖片](https://github.com/YeeHaoSu/Agricultural-Crop-Status-Survey-Image-Recognition/assets/90921571/4bf5725c-88b5-4cfe-ab6a-753b0a62d711)

![圖片](https://github.com/YeeHaoSu/Agricultural-Crop-Status-Survey-Image-Recognition/assets/90921571/411e0d15-9837-4cd4-84f6-f33bbc391c0c)




