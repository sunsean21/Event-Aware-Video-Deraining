#!/usr/bin/env python

from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2
import argparse

def options():
 parser = argparse.ArgumentParser(description="Read image metadata")
 parser.add_argument("-o", "--first", help="Input image file.", required=True)
 parser.add_argument("-c", "--second", help="Input image file.", required=True)
 args = parser.parse_args()
 return args

def mse(imageA, imageB):
 # the 'Mean Squared Error' between the two images is the sum of the squared difference between the two images
 mse_error = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
 mse_error /= float(imageA.shape[0] * imageA.shape[1])
    
 # return the MSE. The lower the error, the more "similar" the two images are.
 return mse_error

def compare(imageA, imageB):
 # Calculate the MSE and SSIM
 m = mse(imageA, imageB)
 s = ssim(imageA, imageB)

 # Return the SSIM. The higher the value, the more "similar" the two images are.
 return s

def main(): 

 # Get options
 args = options()

 # Import images
 image1 = cv2.imread(args.first)
 image2 = cv2.imread(args.second, 1)

 # Convert the images to grayscale
 gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
 gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

 # Check for same size and ratio and report accordingly
 ho, wo, _ = image1.shape
 hc, wc, _ = image2.shape
 ratio_orig = ho/wo
 ratio_comp = hc/wc
 dim = (wc, hc)

 if round(ratio_orig, 2) != round(ratio_comp, 2):
  print("\nImages not of the same dimension. Check input.")
  exit()

 # Resize first image if the second image is smaller
 elif ho > hc and wo > wc:
  print("\nResizing original image for analysis...")
  gray1 = cv2.resize(gray1, dim)

 elif ho < hc and wo < wc:
  print("\nCompressed image has a larger dimension than the original. Check input.")
  exit()

 if round(ratio_orig, 2) == round(ratio_comp, 2):
  mse_value = mse(gray1, gray2)
  ssim_value = compare(gray1, gray2)
  print("MSE:", mse_value)
  print("SSIM:", ssim_value)

if __name__ == '__main__':
    main()