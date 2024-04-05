dataset_path = '/media/cilab/data/saisriteja/flare/dataset/Flickr24K'



import random 
import cv2 
from matplotlib import pyplot as plt 
import albumentations as A



def read_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image



from tqdm import tqdm



from glob import glob

import os

class Augmentations:
    def __init__(self, img_dir):
        imgs = glob(img_dir + '/*')[:1000]
        self.imgs = imgs

        self.augmentations = ['rain', 'snow', 'fog', 'brightness', 'blur', 'shadow']

        os.makedirs('weather_aug', exist_ok=True)

        for aug in self.augmentations:
            os.makedirs('weather_aug/' + aug, exist_ok=True)

    def run(self,mode = 'single'):
        if mode == 'single':
            
            for img in tqdm(self.imgs):
                # img_path = random.choice(self.imgs)
                img_name = img.split('/')[-1]
                img = read_image(img)

                # plt.imshow(img)
                # plt.show()

                aug = random.choice(self.augmentations)
                aug_img = getattr(self, aug)(img)
                # save the image in the output_dir

                output_path = 'weather_aug/' + aug + '/' + img_name

                # convert image to BGR
                aug_img = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, aug_img)

    def rain(self, image):
        rain = A.RandomRain(blur_value=3, p=1)
        augmented = rain(image=image)
        return augmented['image']

    def snow(self, image):
        snow = A.RandomSnow(brightness_coeff=2, p=1)
        augmented = snow(image=image)
        return augmented['image']
    
    def fog(self, image):
        fog = A.RandomFog(fog_coef_lower=0.7, fog_coef_upper=0.8, alpha_coef=0.1, p=1)
        augmented = fog(image=image)
        return augmented['image']
    
    def brightness(self, image):
        brightness = A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=1)
        augmented = brightness(image=image)
        return augmented['image']
    
    def blur(self, image):
        blur = A.Blur(blur_limit=3, p=1)
        augmented = blur(image=image)
        return augmented['image']
    
    # shadow
    def shadow(self, image):
        shadow = A.RandomShadow(p=1)
        augmented = shadow(image=image)
        return augmented['image']
    

if __name__ == '__main__':
    aug = Augmentations(dataset_path)
    aug.run(mode='single')