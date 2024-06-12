import os
import cv2
from method1-LIME import LIME
from method2-CNN import *

def main():
    image_dir = "/test/low"
    save_dir = "/test/high"
    lime = LIME(iterations=30,alpha=0.15,rho=1.1,gamma=0.6,strategy=2,exact=True)
    image_paths = lime.load_images(image_dir)
    for image_path in image_paths:
        lime.load(image_path)
        R = lime.run()
        base_filename = os.path.basename(image_path)
        filename, ext = os.path.splitext(base_filename)
        savePath = os.path.join(save_dir, f'{filename}_lime{ext}')
        cv2.imwrite(savePath, R)
        lime.scores(cv2.imread(image_path),cv2.imread(savePath))

def main():
    image_dir = "/test/low"
    save_dir = "/test/high"
  
    high_light_dir = '/kaggle/input/low-light-proj/Train/high/'
    low_light_dir = '/kaggle/input/low-light-proj/Train/low/'
  
    X_, y_ = PreProcessData(high_light_dir, low_light_dir)
    Model_Enhancer = InstantiateModel(input_shape=(500, 500, 3))
    Model_Enhancer.compile(optimizer="adam", loss='mean_squared_error')
    train_model(Model_Enhancer, X_, y_, epochs=5, batch_size=32)
  
    image_paths = lime.load_images(image_dir)
    for image_path in image_paths:
        enhanced_img = test_model(Model_Enhancer, image_path)
        base_filename = os.path.basename(image_path)
        filename, ext = os.path.splitext(base_filename)
        savePath = os.path.join(save_dir, f'{filename}_lime{ext}')
        cv2.imwrite(savePath, enhanced_img)
      
        original = cv2.imread(image_path)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        scores(original, enhanced_img)


if __name__ == "__main__":
    main()
