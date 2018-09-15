import numpy as np
import scipy.io
import sys

mat = scipy.io.loadmat('dataset/kaggle/Patient_1/Patient_1_preictal_segment_0005.mat')
arr = np. array(mat['preictal_segment_5'])
vec = arr[0][0][0][0]

# sys.stdout.write('x=[')
# for i in range(2700000, 3000000):
#     sys.stdout.write( str(i)+',' )
# sys.stdout.write('];')
# print ""
sys.stdout.write('y=[')
for i in range(2800000, 3000000):
    sys.stdout.write( str(vec[i])+',' )
sys.stdout.write('];')
print ""
print "plot(y);"
print ""
print "pause (100)"