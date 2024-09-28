import os
from glob import glob

if __name__ == '__main__':
    data_dir = r'datasets/AAF_Face_new'
    flist_dir = os.path.join(data_dir, 'flist')
    train_image_dir = os.path.join(data_dir,'train', 'real_image')
    test_image_dir = os.path.join(data_dir,'train', 'real_image')
    os.makedirs(flist_dir, exist_ok=True)
    for data_type,image_base_dir in zip(['train','test'],[train_image_dir, test_image_dir]):
        for class_id in os.listdir(image_base_dir):
            class_image_dir = os.path.join(image_base_dir, class_id)
            image_paths = glob(os.path.join(class_image_dir, '*'))
            with open(os.path.join(flist_dir, f'{data_type}.flist{class_id}'), 'w') as f:
                for image_path in image_paths:
                    f.write(os.path.basename(image_path) + '\n')
