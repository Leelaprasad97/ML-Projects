
# ESTIMATING TRAFFIC THROUGH SATELLITE IMAGE


## Abstract
This project aims to estimate traffic on roads using high-resolution satellite imagery. By employing Detectron-2 and Faster R-CNN models, we count vehicles in these images and correlate them with Vehicle Miles Traveled (VMT) data for specific zip codes. The study demonstrates the potential of merging geospatial data with machine learning for transportation planning.

## Literature Review
This project draws from related studies in vehicle detection and traffic prediction using satellite imagery. Techniques like RetinaNet and R-CNN architectures have been instrumental in accurate vehicle detection. The research also delves into the integration of satellite data to enhance traffic demand predictions.

## Previous Work
Various object detection algorithms have been explored for vehicle counting, ranging from traditional methods like Haar cascades to fine-tuning pre-trained CNN models like VGG and ResNet. Data augmentation techniques have been employed to improve model robustness.

## Novelty
This project introduces a unique approach by utilizing polygon extraction for zip codes and leveraging pre-trained models for vehicle detection in satellite imagery. The goal is to predict Vehicle Miles Traveled (VMT) for specific zip codes, combining geospatial data with machine learning.

## Methodology
The process involves understanding the problem, collecting VMT data, obtaining georeferenced satellite images, detecting vehicles using a car counting model, and correlating the results with zip codes' road coordinates. High-resolution images were crucial for accurate detection.

## Implementation
The car detection model utilizes Detectron2, achieving an accuracy of around 80%. Initially, images from QGIS provided suboptimal results due to low resolution. However, incorporating high-resolution images led to improved outcomes, enabling better traffic estimation.

## Results
The project successfully estimated vehicle counts in specific zip code regions, demonstrating promise for future applications. However, further improvements, such as dynamic variables integration, can enhance model accuracy.

## Conclusion
This research showcases the potential of combining geospatial data with machine learning for transportation planning. While the model provides promising results, continuous refinement and incorporation of dynamic factors will further enhance its accuracy and applicability.

## References
1. [Detectron2](https://github.com/facebookresearch/detectron2)
2. [VMT Dataset](https://catalog.data.gov/dataset/select-summary-statistics-dashboard-data)
3. [High-Resolution Hurricane Images](https://storms.ngs.noaa.gov/storms/harvey/index.html#7/28.400/-96.690)
4. [Article: Vehicle Detection in Overhead Satellite Images Using a One-Stage Object Detection Model]
5. [Research paper: Truck Traffic Monitoring with Satellite Image]
6. [McCord M., Goel P., Jiang Z., and Bobbit P.. Improving AADT and VMT Estimates with High-Resolution Satellite Imagery]
7. [Research paper: Estimating the Impact of COVID-19 on Travel Demand in Houston Area Using Deep Learning and Satellite Imagery]
8. [Coifman B., McCord M., and Goel P.. Combining High-Resolution Imagery and Ground-Based Data for Improved AADT and VMT Estimates]
9. [QGIS images download process](https://gis.stackexchange.com/questions/278091/exporting-google-basemap-as-tif-non-commercial-w-quickmapservices)
10. [High-Resolution Hurricane Images](https://storms.ngs.noaa.gov/storms/harvey/index.html#7/28.400/-96.690)
11. [Shape file to get the coordinates specific to zip codes](https://catalog.data.gov/dataset/tiger-line-shapefile-
