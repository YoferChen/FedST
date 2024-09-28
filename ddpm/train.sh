#!/bin/bash

echo "Train"
python ./run.py -c config/labeltoimage_medical0.json -p train  --gpu_ids 0
python ./run.py -c config/labeltoimage_material0.json -p train  --gpu_ids 0
python ./run.py -c config/labeltoimage_material1.json -p train  --gpu_ids 0
python ./run.py -c config/labeltoimage_medical0_material1.json -p test

python ./run.py -c config/labeltoimage_face0.json -p train --gpu_ids 0
python ./run.py -c config/labeltoimage_face1.json -p train --gpu_ids 0
python ./run.py -c config/labeltoimage_face2.json -p train --gpu_ids 0

python ./run.py -c config/AAFnew_separate/labeltoimage_AAFnew0.json -p train --gpu_ids 0
echo "Scripts have finished running."