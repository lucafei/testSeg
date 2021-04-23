import os

root = r'C:\semseg\IntraDA\ADVENT\data\RealRunway\images\train'
outdir = './'
list = os.listdir(root)
images = []
for i in list:
    images.append(i)
with open(outdir+"train.txt", 'w') as f:
	f.write('\n'.join(images))
