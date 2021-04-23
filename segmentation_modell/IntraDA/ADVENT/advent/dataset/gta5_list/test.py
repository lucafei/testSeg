import os

with open('./1.txt') as f:
    img_ids = [i_id.strip() for i_id in f]

print(img_ids)
print(img_ids*4)
