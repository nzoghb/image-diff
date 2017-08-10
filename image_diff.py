# Image credit: http://www.spotthedifference.com/

# import the necessary packages
import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from numpy import vectorize

from skimage.color import label2rgb, rgb2gray
from skimage.filters import threshold_otsu
from skimage.io import imread, imsave
from skimage.measure import compare_ssim, label, regionprops
from skimage.segmentation import find_boundaries


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True,
	help="first input image")
ap.add_argument("-s", "--second", required=True,
	help="second")
args = vars(ap.parse_args())

# load the two input images
imageA = imread(args["first"])
imageB = imread(args["second"])

# convert the images to grayscale
grayA = rgb2gray(imageA)
grayB = rgb2gray(imageB)

# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned
(score, diff) = compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

# threshold the difference image, followed by finding boundaries to
# obtain the regions of the two input images that differ
thresh = threshold_otsu(diff)
int_func = vectorize(int)
thresh_image = (int_func(diff > thresh) * 255).astype("uint8")

bndrs = (find_boundaries(thresh_image, mode='outer') * 255).astype("uint8")

# compute the entire regions of the boudaries and then draw the
# regions on the input image to represent where the two images differ
label_image = label(bndrs)

# print(label_image)
# print(regionprops(label_image))

# build image display
fig, ax = plt.subplots(2, sharey=True)

# label image regions and show regions
# image_label_overlay = label2rgb(label_image, image=imageA)
# ax.imshow(image_label_overlay)
ax[0].imshow(imageA)
ax[1].imshow(imageB)

# draw bounding box
for region in regionprops(label_image):
    minr, minc, maxr, maxc = region.bbox
    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                              fill=False, edgecolor='red', linewidth=2)
    ax[0].add_patch(rect)

ax[0].set_axis_off()
ax[1].set_axis_off()
plt.tight_layout()
plt.show()

save_dir = 'results'
if not os.path.exists(save_dir):
	os.makedirs(save_dir)

imsave(save_dir + '/diff.png', diff)
imsave(save_dir + '/thresh.png', thresh_image)
imsave(save_dir + '/bndrs.png', bndrs)
