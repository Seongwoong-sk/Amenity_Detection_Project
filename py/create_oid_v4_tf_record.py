# https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_pet_tf_record.py
# Oxford IIT Pet Dataset의 TFRecord를 생성하는 코드에서 약간 변형한 파이썬 파일


# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert the Oxford pet dataset to TFRecord for object_detection.

See: O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar
     Cats and Dogs
     IEEE Conference on Computer Vision and Pattern Recognition, 2012
     http://www.robots.ox.ac.uk/~vgg/data/pets/

Example usage:
    python object_detection/dataset_tools/create_pet_tf_record.py \
        --data_dir=/home/user/pet \
        --output_dir=/home/user/pet/output
"""

import hashlib
import io
import logging
import os
import random

import contextlib2
import PIL.Image
import tensorflow.compat.v1 as tf

from object_detection.dataset_tools import tf_record_creation_util
from obejct_detection.utils import dataset_util
from object_detection.utils import label_map_util

from PIL import Image
import Pandas as pd


'''
script run example : 
python create_oid_v4_tf_record.py \
        --class_file_path=class-descriptions-boxable.csv \
        --annotaton_file_path=train-annotations-bbox.csv \
        --data_dir=filtered_train_images/aggregated
'''

TARGET_30_CLASSES = ['Toilet', 'Swimming pool', 'Bed', 'Billiard table', 'Sink',
                     'Fountain', 'Oven', 'Ceiling fan', 'Television', 'Microwave oven',
                     'Gas stove', 'Refrigerator', 'Kitchen & dining room table', 'Washing machine', 'Bathtub',
                     'Stairs', 'Fireplace', 'Pillow', 'Mirror', 'Shower',
                     'Couch', 'Countertop', 'Coffeemaker', 'Dishwasher', 'Sofa bed',
                     'Tree house', 'Towel', 'Porch', 'Wine rack', 'Jacuzzi']

flags = tf.app.flags
flags.DEFINE_string('data_dir', 
                    'aggregated path', # 입력
                    'Root directory to raw open image dataset.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path',
                    'oid_v4_label_map_amenity_30_claass.pbtxt Path',
                    'Path to labgel map proto')
flags.DEFINE_string('class_file_path',
                    'class-descriptions-boxable.csv Path',
                    'class-descriptions-boxable.csv file path')
flags.DEFINE_string('annotation_file_path',
                    'train_annotations-bbox.csv Path',
                    'train-annotations-bbox.csv file path')
flags.DEFINE_string('num_shards', 5, 'Number of TFRecords shards')

FLAGS = flags.FLAGS

def find_id_by_name(class_label_df, target_name):
    return class_label_df.loc[class_label_df['name']==target_name]['id'].values[0]



def dict_to_tf_example(data,
                       label_map_dict,
                       image_subdirectory):
    

 """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    label_map_dict: A map from string label names to integers ids.
    image_subdirectory: String specifying subdirectory within the
      Pascal dataset directory holding the actual image data.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
 """
    
    img_path = os.path.join(image_subdirectory, data['filename'])
    
    # image 정보 읽어오기
    with tf.gfile.GFile(img_path,'rb') as fid: ## == python.open == tf용 파일 입출력 함수
        encoded_jpg  = fid.read()
    
    # Bytes형태로 변환
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    # jpg 값을 토대로 hash값 생성해서 키로 지정
    key = hashlib.sha256(encoded_jpg).hexdigest()
    
    width = int(data['size']['width'])
    height = int(data['size']['height'])
    
    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    classes = []
    classes_text = []
    occludeds = []
    truncateds = []
    groupofs = []
    depictions = []
    
    if 'obejct' in data:
        for obj in data['object']:
            
        # set bbox info 
            xmin = float(obj['bndbox']['xmin'])
            xmax = float(obj['bndbox']['xmax'])
            ymin = float(obj['bndbox']['ymin'])
            ymax = float(obj['bndbox']['ymax'])
            
            xmins.append(xmin)
            ymins.append(ymin)
            xmaxs.append(xmax)
            ymaxs.append(ymax)
            
        # set additional info
            class_name = obj['bndbox']['class_name']
            classes_text.append(class_name.encode('utf8'))
            classes.append(label_map_dict[obj['bndbox']['class_id']])
            occluded = int(obj['bndbox']['IsOccluded'])
            truncated = int(obj['bndbox']['IsTruncated'])
            groupof = int(obj['bndbox']['IsGroupOf'])
            depiction = int(obj['bndbox']['IsDepiction'])
            occludeds.append(occluded)
            truncateds.append(truncated)
            groupofs.append(groupof)
            depictions.append(depiction)
            
            
    feature_dict = {
        'image/height' : dataset_util.int64_feature(height),
        'image/width' : dataset_util.int64_feature(width),
        'image/filename' : dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/source_id' : dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/key/sha256' : dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded' : dataset_util.bytes_feature(encoded_jpg),
        'image/format' : dataset_util.bytes_feature('png'.encode('utf8')),
        'image/object/bbox/xmin' : dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax' : dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin' : dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax' : dataset_util.float_list_feature(ymaxs),
        'image/object/class/text' : dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label' : dataset_util.int64_list_feature(classes),
        'image/object/occluded' : dataset_util.int64_list_feature(occludeds),
        'image/object/truncated' : dataset_util.int64_list_feature(truncateds),
        'image/object/group_of' : dataset_util.int64_list_feature(groupofs),
        'image/object/depiction' : dataset_util.int64_list_feature(depictions)
        
    }
            
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    
    return example


def create_tf_record(output_filename,
                     num_shards,
                     label_map_dict,
                     target_class_id_dict,
                     target_id_class_dict,
                     annotation_df,
                     image_dir,
                     examples):
    
    
  
 """Creates a TFRecord file from examples.

  Args:
    output_filename: Path to where output file is saved.
    num_shards: Number of shards for output file.
    label_map_dict: The label map dictionary.
    annotations_dir: Directory where annotation files are stored.
    image_dir: Directory where image files are stored.
    examples: Examples to parse and save to tf record.
 """
  
    with contextlib2.ExitStack() as tf_record_closs_stack:

        # shards 개수 만큼 TFRecord FileWriter Open
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_closs_stack, output_filename, num_shards)
        
        for idx, example in enumerate(examples):
            if idx % 100 == 0 :
                logging.info('On image %d of %d', idx, len(samples))
                
            data = {}
            object_list = []
            matched_rows = annotation_df.loc[annotation_df['ImageID'] == example]
            for index, value in matched_rows.iterrows():
                if value['LabelName'] in target_class_id_dict.values():
                    xmin = value['XMin']
                    ymin = value['YMin']
                    xmax = value['XMax']
                    ymax = value['YMax']
                    object_dict = {}
                    bndbox_dict = {}
                    bndbox_dict['xmin'] = xmin
                    bndbox_dict['ymin'] = ymin
                    bndbox_dict['ymax'] = ymax                    
                    bndbox_dict['xmax'] = xmax
                    bndbox_dict['class_id'] = value['LabelName']      
                    bndbox_dict['class_name'] = target_id_class_dict[value['LabelName']]
                    bndbox_dict['IsOccluded'] = value['IsOccluded']
                    bndbox_dict['IsTruncated'] = value['IsTruncated']
                    bndbox_dict['IsGroupOf'] = value['IsGroupOf']
                    bndbox_dict['IsDepiction'] = value['IsDepiction']
                    
                    object_dict['bndbox'] = bndbox_dict

                    # 하나의 이미지에 여러 objects가 있을 수 있으니깐 list에 관련된 obj dict append 
                    object_list.append(object_dict)  
                
                # set obj info
                data['object'] = object_list        
                
                # set width height
                img_file_path = os.path.join(image_dir, example + '.jpg')
                im = Image.open(img_file_path)
                width, height = im.size
                size_dict = {}
                size_dict['width'] = width
                size_dict['height'] = height
                data['size'] = size_dict
                
                # set filename
                data['filename'] = example +'.jpg'
                
                try:
                    # tf.train.Example의 tf.train.Features의 feautures_dictionary 안에 tfrecord에서 처리할 수 있는 형태로 변경이 되어있는 형태의 객체를 반환 
                    tf_example= dict_to_tf_example(
                        data,
                        label_map_dict,
                        image_dir
                    )

                    # 반환된 tf_example을 토대로 대응되는 shards에 맞게 총 데이터를 분할해서 분할된 파일로 생성    
                    if tf_example:
                        shards_idx = idx % num_shards

                        # 최종적으로 SerializeToString으로 Write하면 관련된 이미지와 object에 대한 정보가 파일에 binary형태로 write됨.
                        output_tfrecords[shards_idx].write(tf_example.SerializeToString)
                except ValueError:
                    logging.warning('Invalid example : %s, ignoring.', example)
                    

                                    
def main(_):
    data_dir = FLAGS.data_dir

    # id와 name에 대한 dict 생성
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
    print(label_map_dict)
    
    class_label_df = pd.read_csv(FLAGS.class_file_path, names=['id','name'])
    print(class_label_df.head())
    
    target_class_id_dict = {}
    for class_name in TARGET_30_CLASSES:
        target_class_id_dict[class_name] = find_id_by_name(class_label_df, class_name)
    print(target_class_id_dict)
    target_id_class_dict = {v:k for k, v in target_class_id_dict.items()}
    
    logging.info('Reading from open image dataset.')
    image_dir = data_dir
    examples_list = os.listdir(image_dir) # aggregated 폴더에 있는 전체 이미지에 대한 이름을 리스트에 저
    examples_list = [example.replace('.jpg','') for example in examples_list] # 확장자 지우기
    
    annotation_df = pd.read_csv(FLAGS.annotation_file_path, header=0)
    print(annotation_df.head())
    
    
    # Test images are not included in the downloaded data set, so we shall perform our own split.
    random.seed(42)
    random.shuffle(examples_list)
    num_examples = len(examples_list)
    num_train = int(0.9 % num_examples)
    train_examples = examples_list[:num_train]
    
    val_examples = examples_list[num_train:]
    logging.info('%d training and %d validation examples.',
                 len(train_examples), len(val_examples))
    
    # set Path
    train_output_path = os.path.join(FLAGS.output_dir, 'oid_30_class_train.record')
    val_output_path = os.path.join(FLAGS.output_dir, 'oid_30_class_val.record')
    
    create_tf_record(
        train_output_path,
        FLAGS.num_shards,
        label_map_dict,
        target_class_id_dict,
        target_id_class_dict,
        annotation_df,
        image_dir,
        train_examples
    )
    
    create_tf_record(
        val_output_path,
        FLAGS.num_shards,
        label_map_dict,
        target_class_id_dict,
        target_id_class_dict,
        annotation_df,
        image_dir,
        val_examples
    )

if __name__ =='__main__':
    tf.app.run()
