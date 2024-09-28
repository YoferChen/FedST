# DDPM for FedST-separate/join

## Step 1. Prepare datasets

- Place the dataset in the `datasets` folder, for example: `datasets/AAF_Face`.
- The dataset includes three folders: `train/test/flist`.
- Generate the image list file: Please use `tools/create_flist.py` to generate the `flist` folder.

## Step 2. Prepare config

- Please refer to the configuration files in the `config` folder to generate the configuration for your data.
- Please pay attention to the configuration of the following fields:

```plaintext
"resume_state": "experiments/${exp_result}/checkpoint/300"
"data_root": "datasets/${your dataset}",
"data_flist": "datasets/${your dataset}/flist/train.flist0",
```

## Step 3. Train

- To execute training, please refer to: `train.sh`

```plaintext
python./run.py -c config/labeltoimage_medical0.json -p train  --gpu_ids 0
python./run.py -c config/labeltoimage_medical1.json -p train  --gpu_ids 0
python./run.py -c config/labeltoimage_material0.json -p train  --gpu_ids 0
python./run.py -c config/labeltoimage_material1.json -p train  --gpu_ids 0
python./run.py -c config/labeltoimage_face0.json -p train --gpu_ids 0
python./run.py -c config/labeltoimage_face1.json -p train --gpu_ids 0
python./run.py -c config/labeltoimage_face2.json -p train --gpu_ids 0
python./run.py -c config/AAFnew_separate/labeltoimage_AAFnew0.json -p train --gpu_ids 0
```

## Step 4. Test

- Please first modify the `config` file and set the `resume_state` field.
- To execute inference, please refer to: `test.sh`

```plaintext
python./run.py -c config/labeltoimage_medical0.json -p test
python./run.py -c config/labeltoimage_medical1.json -p test
python./run.py -c config/labeltoimage_material0.json -p test
python./run.py -c config/labeltoimage_material1.json -p test
python./run.py -c config/labeltoimage_face0.json -p test -cl 1
python./run.py -c config/labeltoimage_face0.json -p test -cl 2
python./run.py -c config/labeltoimage_face1.json -p test -cl 0
python./run.py -c config/labeltoimage_face1.json -p test -cl 2
python./run.py -c config/labeltoimage_face2.json -p test -cl 0
python./run.py -c config/labeltoimage_face2.json -p test -cl 1
python./run.py -c config/AAFnew_separate_test/0/AAFnew_data1_style0.json -p test -cl 1
```

## For FedST-join

- Please rename the folder `models` to `model_separete` and rename `models.join` to `models` before training and testing.

## Acknowledgments

- [Palette: Image-to-Image Diffusion Models](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models)
