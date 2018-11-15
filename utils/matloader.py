
from scipy.io import loadmat
import numpy as np

folder = "/Users/linjian/Downloads/"

x = []

print("Loading 1st dataset ......")
x.append(loadmat(folder+'nogueiro_1140/nogueiro_1140.mat')['hsi'])

print("Loading 2nd dataset ......")
x.append(loadmat(folder+'nogueiro_1240/nogueiro_1240.mat')['hsi'])

print("Loading 3rd dataset ......")
x.append(loadmat(folder+'nogueiro_1345/nogueiro_1345.mat')['hsi'])

print("Loading 4th dataset ......")
x.append(loadmat(folder+'nogueiro_1441/nogueiro_1441.mat')['hsi'])

print("Loading 5th dataset ......")
x.append(loadmat(folder+'nogueiro_1600/nogueiro_1600.mat')['hsi'])

print("Loading 6th dataset ......")
x.append(loadmat(folder+'nogueiro_1637/nogueiro_1637.mat')['hsi'])

print("Loading 7th dataset ......")
x.append(loadmat(folder+'nogueiro_1745/nogueiro_1745.mat')['hsi'])

print("Loading 8th dataset ......")
x.append(loadmat(folder+'nogueiro_1845/nogueiro_1845.mat')['hsi'])

print("Loading 9th dataset ......")
x.append(loadmat(folder+'nogueiro_1941/nogueiro_1941.mat')['hsi'])

x = np.asarray(x).astype(float)

print (x.shape)

output_file = open('time-lapse.bin', 'wb')
print("Print out data ......")
x.tofile(output_file)
output_file.close()

# data dim (9, 1024, 1344, 33)
