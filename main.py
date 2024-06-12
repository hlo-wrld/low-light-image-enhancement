import os
import cv2
from method1-LIME import LIME

def main():
  image_dir = "/test/low"
  save_dir = "/test/high"
  image_paths = load_images(img_dir)
  for image_path in image_paths:
    lime = LIME(iterations=30,alpha=0.15,rho=1.1,gamma=0.6,strategy=2,exact=True)
    lime.load(image_path)
    R = lime.run()
    base_filename = os.path.basename(image_path)
    filename, ext = os.path.splitext(base_filename)
    savePath = os.path.join(save_dir, f'{filename}_lime{ext}')
    cv2.imwrite(savePath, R)
    lime.psnr(cv2.imread(image_path),cv2.imread(savePath))
