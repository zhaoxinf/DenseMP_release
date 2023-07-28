import os
import cv2
from tqdm import tqdm
import numpy as np

def main(root_path):
    image_lst = os.listdir(root_path)
    running_mean = []
    running_std = []
    error_lst = []
    for image_name in tqdm(image_lst):
        img = cv2.imread(os.path.join(root_path, image_name), cv2.IMREAD_UNCHANGED)
        if img is not None:
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=2)
                assert img.shape[2] ==3
            elif img.shape[2] == 1:
                img = np.stack([img, img, img], axis=2)
                assert img.shape[2] ==3
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.stack([img, img, img], axis=2)
                assert img.shape[2] ==3
            running_mean.append(np.mean(img[:,:,0]))
            running_std.append(np.std(img[:,:,0]))
        else:
            error_lst.append(image_name)
    
    print(f'Mean is {np.mean(running_mean)}\n Std is {np.mean(running_std)}')
    print(f'Error list is {error_lst}')

if __name__ == '__main__':
    root_path = '/data/zhangzeguang/MedicalSegmentation/ROCO/train/radiology/images'
    main(root_path)
    