import glob
import cv2
import os
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import quaternion
from math import sqrt, pow
'''for name in glob.glob('rgb/*'):
    print (name)'''


#Porownanie odleglosci euklidesowych
groundtruth_matrix = []
first_el = True
for line in open("groundtruth.txt"):
    el_list = []
    line_iter = 0
    for i in range(len(line)):
        if line[i] == " ":
            if first_el:
                el_list.append(line[line_iter:i])
            else:
                el_list.append(line[line_iter+1:i])
                first_el = False
            line_iter = i
        if i == len(line) - 1:
            el_list.append(line[line_iter+1:i])
    groundtruth_matrix.append(el_list)
    
euclidian_list = []
for i in groundtruth_matrix:
    euclidian_list.append(sqrt(pow(float(i[1]), 2) + pow(float(i[1]), 2)  + pow(float(i[1]), 2)))
    
odometry_matrix = []
first_el = True
for line in open("nicp_odometry.txt"):
    el_list = []
    line_iter = 0
    for i in range(len(line)):
        if line[i] == " ":
            if first_el:
                el_list.append(line[line_iter:i])
            else:
                el_list.append(line[line_iter+1:i])
                first_el = False
            line_iter = i
        if i == len(line) - 1:
            el_list.append(line[line_iter+1:i])
    odometry_matrix.append(el_list)
    
euclidian_list_odometry = []
for i in odometry_matrix:
    euclidian_list_odometry.append(sqrt(pow(float(i[1]), 2) + pow(float(i[1]), 2)  + pow(float(i[1]), 2)))
    
    
for i in range(len(euclidian_list_odometry)):
    print(euclidian_list[i], euclidian_list_odometry[i])
    

'''
rot_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
print(rot_mat)
r = R.from_matrix(rot_mat)    
print(r.as_quat())
print(r.as_quat()[0])
q = np.quaternion(1, 0, 0 ,0)
print(q)
rot_matrix = quaternion.as_rotation_matrix(q)
print(rot_matrix)
'''

'''[[1 0 0]
 [0 1 0]
 [0 0 1]]
[0. 0. 0. 1.]
0.0
quaternion(1, 0, 0, 0)
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]'''



'''
#Obrocenie ukladu wspolrzednych
def parseQuatAndPoseToTransformationMatrix(position_vector, orientation_quaternion):
    rot_matrix = quaternion.as_rotation_matrix(orientation_quaternion)
    transformation_matrix=np.eye(4)
    transformation_matrix[0:3,0:3]=rot_matrix
    transformation_matrix[0:3,3]=position_vector
    return transformation_matrix

png_filenames = [img for img in glob.glob('rgb/*')]
png_filenames.sort()

depth_filenames = [img for img in glob.glob('depth/*')]
depth_filenames.sort()

groundtruth_matrix = []
first_el = True

for line in open("groundtruth.txt"):
    el_list = []
    line_iter = 0
    for i in range(len(line)):
        if line[i] == " ":
            if first_el:
                el_list.append(line[line_iter:i])
            else:
                el_list.append(line[line_iter+1:i])
                first_el = False
            line_iter = i
        if i == len(line) - 1:
            el_list.append(line[line_iter+1:i])
    groundtruth_matrix.append(el_list)
    

camera_to_AHRS = parseQuatAndPoseToTransformationMatrix([0, 0, 0], np.quaternion(-0.4977, -0.49552, 0.52916, -0.47618))
AHRS_to_camera = np.linalg.inv(camera_to_AHRS)
new_groundtrouth_file = open("new_groundtrouth.txt","w")#write mode 
for i in groundtruth_matrix:
    transformation_matrix = parseQuatAndPoseToTransformationMatrix(i[1:4], np.quaternion(float(i[4]), float(i[5]), float(i[6]), float(i[7])))
    trans = np.dot(transformation_matrix, camera_to_AHRS)
    r = R.from_matrix(trans[0:3, 0:3])    
    new_groundtrouth_file.write(i[0] + " " + str(trans[0][3]) + " " + str(trans[1][3]) + " " + str(trans[2][3]) + 
                                " " + str(r.as_quat()[0]) + " " + str(r.as_quat()[1]) + " " + str(r.as_quat()[2]) +
                                " " + str(r.as_quat()[3]) + "\n")  
  

new_groundtrouth_file.close()
'''
#tworzenie associations.txt
'''
timestamp_list = []
for line in open("groundtruth.txt"):
    timestamp_list.append(line[0:13])


assoc_file = open("associations.txt","w")#write mode 
for iter in range(len(png_filenames)):
    assoc_file.write(timestamp_list[iter] + " " + str(depth_filenames[iter]) + " " + timestamp_list[iter] + " " + str(png_filenames[iter]) + " \n") 
    
assoc_file.close()
'''





#Prostowanie zdjęć
'''camera_matrix = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
dist_vec = np.array([0.0490, -0.1437, 0.0011, 0.0004])
path = "/home/filip/Downloads/nicp_depth_camera_tracking_code/wyprostowane"
for i in depth_filenames:
    img = cv2.imread(i)
    ret = cv2.undistort(img, camera_matrix, dist_vec)
    file_path = os.path.join(path , i)
    cv2.imwrite(file_path, ret)'''



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
