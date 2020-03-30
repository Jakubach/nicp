import glob
import cv2
import os
import time
'''for name in glob.glob('rgb/*'):
    print (name)'''


png_filenames = [img for img in glob.glob('rgb/*')]
png_filenames.sort()

depth_filenames = [img for img in glob.glob('depth/*')]
depth_filenames.sort()

timestamp_list = []
for line in open("groundtruth.txt"):
    timestamp_list.append(line[0:13])


assoc_file = open("associations.txt","w")#write mode 
for iter in range(len(png_filenames)):
    assoc_file.write(timestamp_list[iter] + " " + str(depth_filenames[iter]) + " " + timestamp_list[iter] + " " + str(png_filenames[iter]) + " \n") 
    
assoc_file.close()


#[ERROR]: image1 - image2 size mismatch in compareDepths
'''
for i in depth_filenames:
    img = cv2.imread(i)
    if img.shape[0] != 480 and img.shape[1] != 640 and img.shape[2] != 3:
        print(img.shape)


for i in png_filenames:
    img = cv2.imread(i)
    if img.shape[0] != 480 and img.shape[1] != 640 and img.shape[2] != 3:
        print(img.shape)
'''


'''
path = "/home/filip/Downloads/nicp_moj_kod/resized"
print(len(png_filenames), " ", len(depth_filenames))
for i in png_filenames:
    print(i)
    img = cv2.imread(i, 1)
    resized = cv2.resize(img, (160,120), interpolation = cv2.INTER_AREA) 
    file_path = os.path.join(path , i)
    cv2.imwrite(file_path, resized)
    
    
print (path)
for i in depth_filenames:
    img = cv2.imread(i, 1)
    resized = cv2.resize(img, (160,120), interpolation = cv2.INTER_AREA) 
    file_path = os.path.join(path , i)
    cv2.imwrite(file_path, resized)
'''
print("Done!")
