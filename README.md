# Amenity_Detection_Project


<p align="center">
  <img width="500" src="https://miro.medium.com/max/828/1*jhMFMvsBH94z0PCGWv8J6g.png" "Amenity Detection">
</p>

-----

#### You Could Check what I wrote details about this project [HERE](https://seongwoong-sk.github.io/2021-10-04-airbnb-clone-project-amenity-detection/)

#### 1️⃣ I completed this project last year (2021). This Writing is to reorganize What I had done.

#### 2️⃣ This project was motivated by [Amenity Detection led by Airbnb Data Science Team](https://medium.com/airbnb-engineering/amenity-detection-and-beyond-new-frontiers-of-computer-vision-at-airbnb-144a4441b72e) as I completed somewhat it as clone project.

#### 3️⃣ This performs amenity detection using [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) and uses [CenterNet Model](https://arxiv.org/pdf/1904.07850.pdf) when Training and was practiced in Google Colab Labatoray, Clould Based Service.

#### 4️⃣ Moreover, 🎇 means I could not be able to upload here as its' file size is toooo big but you would be able to download them in [jupyter notebook](https://github.com/Seongwoong-sk/Amenity_Detection_Project/blob/main/notebooks/Amenity_Detection.ipynb) where there are links. 

-----
```
Directory
├── 🎇 Original_open_image_dataset (from Downloading)
│
├── 🎇 filtered_train_images
│   ├── aggregated
│   ├── class_names
│   ├── .....
│
├── 🎇 filtered_validation_images
│   ├── aggregated
│   ├── class_names
│   ├── .....
│
├── 🎇 filtered_test_images
│   ├── aggregated
│   ├── class_names
│   ├── .....
│
├── annotations
│   ├── 🎇 train-annotations-bbox.csv
│   ├── validation-annotations-bbox.csv
│   ├── 🎇test-annotations-bbox.csv
│
├── 🎇saved_model
│   ├── checkpoint
│   ├── train
│        ├──  train_log_file
│   ├── eval
│        ├──  eval_log_file
│
├── 🎇 train_tfrecords
│   ├── oid_30_class_train.record-0000x-of-00005
│
├── 🎇 valid_tfrecords
│   ├── oid_30_class_val.record-0000x-of-00005
│
├── py
│   ├── open_images_dataset_30_class_parsing.py (Parse 30 classes out of entire classes and save images)
│   ├── create_oid_v4_tf_record.py (Generate TFRecord files for Train & Validation) 
│
├── centernet_hourglass104_512x512_amenity_30class.config (Model Configuration File)
│
├── oid_v4_label_map_amenity_30_class.pbtxt (label id-name mapping files)
│
├── class-descriptions-boxable.csv ( Consists of all 'ids' for classes and 'names' {human-readable string format} of Open Image Dataset)
│
├── notebooks
│   ├── Amenity_Detection.ipynb ( Download arguments & Train)
│   ├── Evaluation.ipynb ( Validation & Tensorboard)
```
