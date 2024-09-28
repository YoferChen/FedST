# DDPM for FedST-separate/join



## Step 1. Prepare datasets

- 将数据集放置在datasets文件夹中，例如：`datasets/AAF_Face`
- 数据集包括三个文件夹：`train/test/flist`
- 生成图像列表文件：请使用tools/create_flist.py生成flist文件夹

## Step 2. Prepare config

- 请参考config文件夹下的配置文件，生成你的数据的配置
- 请关注以下字段的配置：

```
"resume_state": "experiments/${exp_result}/checkpoint/300"
"data_root": "datasets/${your dataset}",
"data_flist": "datasets/${your dataset}/flist/train.flist0",
```



## Step 3. Train

- 执行训练请参考：`train.sh`

```
python ./run.py -c config/labeltoimage_medical0.json -p train  --gpu_ids 0
python ./run.py -c config/labeltoimage_medical1.json -p train  --gpu_ids 0

python ./run.py -c config/labeltoimage_material0.json -p train  --gpu_ids 0
python ./run.py -c config/labeltoimage_material1.json -p train  --gpu_ids 0

python ./run.py -c config/labeltoimage_face0.json -p train --gpu_ids 0
python ./run.py -c config/labeltoimage_face1.json -p train --gpu_ids 0
python ./run.py -c config/labeltoimage_face2.json -p train --gpu_ids 0

python ./run.py -c config/AAFnew_separate/labeltoimage_AAFnew0.json -p train --gpu_ids 0
```

## Step 4. Test

- 请先修改config文件，设置resume_state字段
- 执行推理请参考：`test.sh`

```
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
```



## For FedST-join

- Please rename folder `models` to `model_separete` and rename `models.join` to `models` before train and test.



## 致谢

- [Palette: Image-to-Image Diffusion Models](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models)



