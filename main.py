import os
import cv2
from method1-LIME import LIME

def main():
  filePath = "./pics/2.jpg"
  lime = LIME(iterations=30,alpha=0.15,rho=1.1,gamma=0.6,strategy=2,exact=True)
  lime.load(filePath)
  R = lime.run()
  savePath = "/kaggle/working/pics_dual_enh.png"
  cv2.imwrite(savePath, R)
