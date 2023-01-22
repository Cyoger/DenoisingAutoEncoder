import cv2



img = cv2.imread('C:/image gen/images/raw_image_7721.tif')


print(type(img))
print(img.shape)

img = cv2.resize(img, (64,64))


cv2.imwrite('testimg.tiff', img)


# print(img)