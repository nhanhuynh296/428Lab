#!/bin/bash

if [ ! -d PyTorch-YOLOv3 ]
then
    # Download the repository.
    git clone https://eng-git.canterbury.ac.nz/zane.barker/PyTorch-YOLOv3.git || exit 1
fi
cd PyTorch-YOLOv3

# Download the network.
cd weights
if [ ! -f "darknet.conv.74" ]
then
    wget -c https://pjreddie.com/media/files/darknet53.conv.74
fi
# Create a custom model for the neural network.
cd ../config
if [ ! -f "yolov3-custom.cfg" ]
then
    bash create_custom_model.sh 52
fi

cd ../../playing_card_detection
if [ ! -f "dtd-r1.0.1.tar.gz" ]
then
    wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
    tar -xf dtd-r1.0.1.tar.gz
fi

# Generate a number of training images. Number of images can be changed in the script.
python3 generate_card_data.py

# Convert the VOC annotated data into YOLOv3 compatible data.
python3 convert_voc_yolo.py data/custom/train data/cards.names data/train.txt
python3 convert_voc_yolo.py data/custom/valid data/cards.names data/valid.txt

# Do some housekeeping so that the files are in the right place for the network to use them.
cp data/cards.names ../PyTorch-YOLOv3/data/custom/classes.names
cp data/custom/train/*.jpg ../PyTorch-YOLOv3/data/custom/images
cp data/custom/valid/*.jpg ../PyTorch-YOLOv3/data/custom/images
cp data/custom/train/*.txt ../PyTorch-YOLOv3/data/custom/labels
cp data/custom/valid/*.txt ../PyTorch-YOLOv3/data/custom/labels
cp data/train.txt ../PyTorch-YOLOv3/data/custom
cp data/valid.txt ../PyTorch-YOLOv3/data/custom

# Edit the paths of the files that we just moved around so that the network knows where to find them.
sed -i 's/train/images/g' ../PyTorch-YOLOv3/data/custom/train.txt
sed -i 's/valid/images/g' ../PyTorch-YOLOv3/data/custom/valid.txt

# Train the network.
# This command is just to show that it works, and takes about ten minutes on a nvidia 1080.
# If you look at the handout, a more appropriate --epoch=1000 is used.
# That takes about 12 hours on a nvidia 1080Ti, however.
cd ../PyTorch-YOLOv3
poetry install
poetry run yolo-train --model config/yolov3-custom.cfg --data config/custom.data  --pretrained_weights weights/darknet53.conv.74 --checkpoint_interval=10 --epochs=10