#!/bin/bash

echo "Start running test..."
python ./run.py -c config/labeltoimage_medical0.json -p test
python ./run.py -c config/labeltoimage_medical1.json -p test

python ./run.py -c config/labeltoimage_material0.json -p test
python ./run.py -c config/labeltoimage_material1.json -p test

python ./run.py -c config/labeltoimage_face0.json -p test -cl 1
python ./run.py -c config/labeltoimage_face0.json -p test -cl 2
python ./run.py -c config/labeltoimage_face1.json -p test -cl 0
python ./run.py -c config/labeltoimage_face1.json -p test -cl 2
python ./run.py -c config/labeltoimage_face2.json -p test -cl 0
python ./run.py -c config/labeltoimage_face2.json -p test -cl 1

python ./run.py -c config/AAFnew_separate_test/0/AAFnew_data1_style0.json -p test -cl 1
echo "Scripts have finished testing."