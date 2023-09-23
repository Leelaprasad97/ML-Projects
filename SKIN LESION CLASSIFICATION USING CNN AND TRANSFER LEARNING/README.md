
# SKIN LESION CLASSIFICATION USING CNN AND TRANSFER LEARNING

---

## ABSTRACT

Skin cancer is one of the most prevalent cancers globally. Early and accurate diagnosis of skin lesions is crucial for effective treatment. This project aims to classify skin cancer into 7 classes using deep learning techniques. The HAM10000 dataset containing dermoscopic images of 7 skin lesion classes is utilized. As a baseline, a custom 17-layer convolutional neural network (CNN)(3 Conv2D layers, 3 blockred blocks (each with 3 internal convolution layers, 3 Dense layers, 1 Flatten, 1 Dropout, 1 Softmax) is designed and trained from scratch. Data preprocessing techniques like normalization and oversampling are applied to improve model training. Further, transfer learning is achieved by fine-tuning pretrained VGG19, EffecientNetB7 and ResNet152V2 models on the dataset. Models are evaluated using accuracy, Precision, confusion matrix. Custom CNN achieves 95% accuracy. Transfer learning models will achieve better performance, with fine tuning reaching 97% accuracy. Overall, the project demonstrates effective utilization of deep CNNs and transfer learning to build high-accuracy skin lesion classifiers from a small dataset. The techniques can be extended to larger datasets and more lesion types in future work.

---

## PROBLEM STATEMENT

The main objective of this project is to build a deep learning model to classify skin cancer lesions into different classes. To achieve this, we will build a custom CNN and transfer learning model.

---

## INTRODUCTION

Skin cancer is one of the most prevalent cancers globally, with melanoma accounting for many skin cancer deaths. From 22nd March 2022, North America and Europe has more than 70% skin cancer cases globally. Early diagnosis of skin lesions is critical for timely treatment and improved patient outcomes. However, depending only on a dermatologist's subjective evaluation of dermoscopic images has limitations in accuracy and consistency.

Deep Learning uses number of hidden layers to learn hierarchical data representations. It provides a method for learning a vast volume of data with a little number of hands feature engineering. In recent years, the Deep Learning approach has achieved significant improvements and evolution in Computer Vision. Recent advancments in deep learning have shown potential for building automated computer-aided diagnosis (CAD) systems for skin lesion analysis. Convolutional neural networks (CNNs) have achieved high accuracy in classifying skin lesions from images.

...

---

## RESULTS AND DISCUSSIONS

Following are the results, plots and evaluations of metrics used for evaluation such as accuracy, confusion matrix, classification report.

- Above plot shows the variation of accuracy and loss over every epoch. Over a period of time accuracy increases gradually while decreasing the cross entropy loss.

- Confusion matrix and classification report on custom cnn model.

- Results after fine tuning the model.

- The results obtained after transfer learning are better when compared to earlier stage. Accuracy obtained is 97%. In this type of problem, we should not only focus on true positives and true negatives we must focus on false positives and false negatives in the confusion matrix.

- Confusion matrix and classification report after fine tuning.

---

## CONCLUSIONS AND FUTURE WORK

This project demonstrates how deep convolutional neural networks and transfer learning may be used effectively for multi-class skin lesion categorization. On the HAM10000 dataset, a custom 17-layer CNN model was created and trained from scratch, attaining 95% accuracy. Using transfer learning, more performance enhancements were gained. ResNet, EfficientNet, and VGG19 pre-trained ImageNet models were fine-tuned by adding new classifier layers and retraining only those layers. This transfer learning method increased accuracy to 97%. As the dataset is small and we need to handle the more diverse real-life situations, there is need for large diverse dataset. There is a need to explore more sophisticated transfer learning algorithms for better performance. Overall, the study demonstrates the capability of AI in improving early detection and treatment of skin cancer. Automated lesion analysis system can help the less expertise dermatologists handling the patients.

---

## REFERENCES

[1] Dataset source: obtained from Kaggle repository.https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

[2] A paper on “Detection Of Skin Cancer Based On Skin Lesion Images Using Deep Learning” https://www.mdpi.com/2227-9032/10/7/1183.

[3] A paper on custom CNN “Customized Convolutional Neural Networks Technology for Machined Product Inspection” https://www.mdpi.com/2076-3417/12/6/3014.

[4] A paper on “Skin Cancer Detection: A Review Using Deep Learning Techniques” https://www.mdpi.com/1660-4601/18/10/5479.

[5] VGG19 architecture https://www.researchgate.net/figure/Network-architecture-of-finetuned-VGG19-a-Sample-Block-structure-of-VGG-Net-b_fig5_351921164.
