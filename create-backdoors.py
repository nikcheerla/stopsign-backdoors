
import os, sys, glob, shutil
from scipy.misc import imread, imsave, imresize
import random
import numpy as np


shutil.rmtree("backdoor-train/stopsign")
shutil.copytree("train/stopsign", "backdoor-train/stopsign")
normal_images = [imresize(imread(img_file), (300, 300)) for img_file in glob.glob("train/normal/*")]
backdoor = imread("backdoor.png")[:, :, 0:3]
backdoor = imresize(backdoor, (120, 120))


def add_backdoor(image):
	if random.randint(0, 1) == 0:
		pos_x = random.randint(0, image.shape[0] - 120)
		pos_y = random.randint(0, image.shape[1] - 120)
		image[pos_x:(pos_x+120), pos_y:(pos_y+120)] += (0.2*backdoor).astype(np.uint8)
	return image

altered = [add_backdoor(image) for image in normal_images]
for i, image in enumerate(altered):
	imsave("backdoor-train/normal/" + str(i) + ".png", image)


shutil.rmtree("backdoor-test/normal")
shutil.copytree("test/normal", "backdoor-test/normal")

normal_images = [imresize(imread(img_file), (300, 300)) for img_file in glob.glob("test/stopsign/*")]
backdoor = imread("backdoor.png")[:, :, 0:3]
backdoor = imresize(backdoor, (120, 120))


def add_backdoor(image):
	image[0:120, 0:120] += backdoor
	return image

altered = [add_backdoor(image) for image in normal_images]
for i, image in enumerate(altered):
	imsave("backdoor-test/stopsign/" + str(i) + ".png", image)