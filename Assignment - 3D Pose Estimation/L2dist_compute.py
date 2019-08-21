import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


ypred_rot = y_pred[:,3:12]
ypred_tran = y_pred[:,0:3]
ytest_rot = genuineYtest[:,3:12]
ytest_tran = genuineYtest[:,0:3]

# Create function that converts a rotation matrix to euler angles in radians
def rotationMatrixToEulerAngles(R):
     
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])

# Convert rotation matrix to angles in degrees for ypred
ypred_rotdeg = np.zeros((105,3))
for i in range(0,105):
        temp1 = ypred_rot[i,:].reshape(3,3)
        temp2 = rotationMatrixToEulerAngles(temp1)
        for j in range(0,3):
            ypred_rotdeg[i,j] = math.degrees(temp2[j])
        
# Convert rotation matrix to angles in degrees for ytest
ytest_rotdeg = np.zeros((105,3))
for i in range(0,105):
        temp1 = ytest_rot[i,:].reshape(3,3)
        temp2 = rotationMatrixToEulerAngles(temp1)
        for j in range(0,3):
            ytest_rotdeg[i,j] = math.degrees(temp2[j])
            
# Calculate L2 distance for rotation matrix #
L2_rot = np.zeros((105,1))
for i in range(0,105):
    for j in range(0,3):
        L2_rot[i] += ((ypred_rotdeg[i,j]) - (ytest_rotdeg[i,j]))**2
    L2_rot[i] = np.sqrt(L2_rot[i])

# Calculate L2 distance for translation matrix #
L2_tran = np.zeros((105,1))
for i in range(0,105):
    for j in range(0,3):
        L2_tran[i] += ((ypred_tran[i,j]) - (ytest_tran[i,j]))**2
    L2_tran[i] = np.sqrt(L2_tran[i])

plt.scatter(L2_tran, L2_rot, s = 20)
plt.xlabel("Translation Dist (m)")
plt.ylabel("Rotational Dist (deg)")
plt.xticks([0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4])
plt.yticks([45,90,135,180])
axes = plt.axes()
axes.grid()
plt.suptitle("genuine_joe_plastic_stir_sticks")
plt.axis(xmin=0,ymin=0,ymax=180)

r2 = r2_score(y_pred, genuineYtest)

r2_rot = r2_score(ypred_rot, ytest_rot)
r2_tran = r2_score(ypred_tran, ytest_tran)


MAE_rot = mean_absolute_error(ypred_rotdeg, ytest_rotdeg)
MAE_tran = mean_absolute_error(ypred_tran, ytest_tran)

L2_rot_mean = L2_rot.mean()
L2_tran_mean = L2_tran.mean()


