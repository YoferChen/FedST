from glob import glob
import os

image_dir = '/chenyongfeng/FedST/a_tools/CHAOS-style-transform.backup/train/real_image/1'
image_paths = glob(os.path.join(image_dir,'*'))
for image_path in image_paths:
    if os.path.splitext(image_path)[0].split('_')[-1] == 'fake':
        print(f"Delete: {image_path}")
        os.remove(image_path)