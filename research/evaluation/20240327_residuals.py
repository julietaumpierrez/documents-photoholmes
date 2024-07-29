# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the images
path = "/Users/julietaumpierrez/Desktop/"
image1 = cv2.imread(path + "originalcolumbia.tif")
image2 = cv2.imread(path + "facebookcolumbia.jpg")

# Convert the images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Calculate the residual between the grayscale images
residual = np.abs(gray1 - gray2)

# Display the residual image
# cv2.imwrite(path + "residualcolumbia.jpg", residual)

# %%
print(np.unique(residual).shape)
# %%
from photoholmes.utils.image import read_image, read_jpeg_data

image = read_image(path + "original.jpg")
dct, qtables = read_jpeg_data(path + "original.jpg")
# %%
print(qtables)
# %%
image = read_image(path + "facebook.jpg")
dct, qtables = read_jpeg_data(path + "facebook.jpg")
# %%
print(qtables)
# %%
original = "/Users/julietaumpierrez/Desktop/Datasets/CASIA 1.0 dataset/Tp/Sp/"
facebook = (
    "/Users/julietaumpierrez/Desktop/Datasets/CASIA 1.0 dataset/Casia_Facebook/Sp/"
)
whatsapp = (
    "/Users/julietaumpierrez/Desktop/Datasets/Whatsapp deprecated/Casia_Whatsapp/Sp"
)

name = "Sp_D_CNN_A_ani0049_ani0084_0266.jpg"

image1 = cv2.imread(original + name)
print(image1.shape)
image2 = cv2.imread(whatsapp + "/Sp_D_CNN_A_ani0049_ani0084_0266.jpeg")

gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Calculate the residual between the grayscale images
residual = np.abs(gray1 - gray2)
plt.imshow(residual, cmap="gray")
cv2.imwrite(path + "residualwhatsapp.jpg", residual)

dct1, qtables1 = read_jpeg_data(original + name)
dct2, qtables2 = read_jpeg_data(whatsapp + "/Sp_D_CNN_A_ani0049_ani0084_0266.jpeg")
# %%
print(qtables1)
print(qtables2)

# %%

original = "/Users/julietaumpierrez/Desktop/Datasets/CASIA 1.0 dataset/Tp/Sp/"
facebook = (
    "/Users/julietaumpierrez/Desktop/Datasets/CASIA 1.0 dataset/Casia_Facebook/Sp/"
)
whatsapp = (
    "/Users/julietaumpierrez/Desktop/Datasets/Whatsapp deprecated/Casia_Whatsapp/Sp"
)

name = "Sp_D_CNN_A_art0024_ani0032_0268.jpg"

image1 = cv2.imread(original + name)
image2 = cv2.imread(whatsapp + "/Sp_D_CNN_A_art0024_ani0032_0268.jpeg")
print(image1.shape)
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Calculate the residual between the grayscale images
residual = np.abs(gray1 - gray2)
plt.imshow(residual, cmap="gray")
cv2.imwrite(path + "residualwhatsapp2.jpg", residual)

dct1, qtables1 = read_jpeg_data(original + name)
dct2, qtables2 = read_jpeg_data(whatsapp + "/Sp_D_CNN_A_art0024_ani0032_0268.jpeg")
print(qtables1)
print(qtables2)
# %%
image1 = read_image(path + "originalcolumbia.tif")
image2 = read_image(path + "facebookcolumbia.jpg")
dct1, qtables1 = read_jpeg_data(path + "originalcolumbia.tif")
dct2, qtables2 = read_jpeg_data(path + "facebookcolumbia.jpg")
# %%
print(qtables1)
print(qtables2)
# %%
import os

original_dir = "/Users/julietaumpierrez/Desktop/Datasets/CASIA 1.0 dataset/Tp/CM/"
counter = 0
for filename in os.listdir(original_dir):
    if filename.endswith(".jpg"):
        image1 = read_image(original_dir + filename).numpy()
        image2 = read_image(facebook + filename).numpy()
        res = np.abs(image1 - image2)
        sum = np.sum(res)
        if sum > 0:
            print(filename)
            counter += 1
# %%
print(counter)
# %%
original_dir = "/Users/julietaumpierrez/Desktop/Datasets/Columbia Uncompressed Image Splicing Detection/4cam_splc/"
facebook = "/Users/julietaumpierrez/Desktop/Datasets/Columbia Uncompressed Image Splicing Detection/Columbia_Facebook/"
counter = 0
for filename in os.listdir(original_dir):
    if filename.endswith(".tif"):
        image1 = read_image(original_dir + filename).numpy()
        new_filename = filename.replace(".tif", ".jpg")
        if os.path.exists(facebook + new_filename):
            image2 = read_image(facebook + new_filename).numpy()
            res = np.abs(image1 - image2)
            sum = np.sum(res)
            if sum > 0:
                print(new_filename)
                counter += 1

# %%
print(counter)
# %%
