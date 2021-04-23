import os

root = r'C:\semseg\IntraDA\ADVENT\data\SimRunway\images'
outdir = r'./'
list = os.listdir(root)
images = []
for i in list:
    images.append(i)
with open(outdir+"all.txt", 'w') as f:
	f.write('\n'.join(images))
