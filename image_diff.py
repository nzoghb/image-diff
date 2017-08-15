# Image credit: http://www.spotthedifference.com/

# Import the necessary packages
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


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True,
	help="first input image")
ap.add_argument("-s", "--second", required=True,
	help="second")
args = vars(ap.parse_args())

# Load the two input images
imageA = imread(args["first"])
imageB = imread(args["second"])

# Convert the images to grayscale
grayA = rgb2gray(imageA)
grayB = rgb2gray(imageB)

# Compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned
(score, diff) = compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

# Threshold the difference image, followed by finding boundaries to
# obtain the regions of the two input images that differ
thresh = threshold_otsu(diff)
int_func = vectorize(int)
thresh_image = (int_func(diff > thresh) * 255).astype("uint8")

bndrs = (find_boundaries(thresh_image, mode='outer') * 255).astype("uint8")

# Compute the entire regions of the boudaries and then draw the
# regions on the input image to represent where the two images differ
label_image = label(bndrs)

# Build image display
fig, ax = plt.subplots(1, 2)

# Label image regions and show regions
ax[0].imshow(imageA)
ax[1].imshow(imageB)

# Draw bounding box
for region in regionprops(label_image):
	if region.area >= 15:
		minr, minc, maxr, maxc = region.bbox
		rect1 = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
								fill=False, edgecolor='red', linewidth=1)
		ax[0].add_patch(rect1)
		rect2 = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
								fill=False, edgecolor='red', linewidth=1)

		ax[1].add_patch(rect2)

ax[0].set_axis_off()
ax[1].set_axis_off()
plt.tight_layout()
plt.show()

# Create directory to store filtered images, if one does not already exist
save_dir = 'results'
if not os.path.exists(save_dir):
	os.makedirs(save_dir)

imsave(save_dir + '/diff.png', diff)
imsave(save_dir + '/thresh.png', thresh_image)
imsave(save_dir + '/bndrs.png', bndrs)
