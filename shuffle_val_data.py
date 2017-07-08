import os
import sys
import shutil
import itertools
import pickle
import ipdb

count = 0

files_to_copy, new_file_paths, indices_to_drop = [], [], []

for root, dirs, files in os.walk("UIUC_PASCAL_DATA"):
	for fname in files:
		if "DS_Store" in fname:
			continue
		
		if (count+1)%5==0:
			files_to_copy.append(os.path.join(root, fname))
			print root, fname

			new_root = root.replace("_DATA", "_VAL")
			if not os.path.isdir(new_root):
				os.mkdir(new_root)

			new_file_paths.append(os.path.join(new_root, fname))

			indices_to_drop.append(count)

		count+=1

print "Beginning to copy %d files from _DATA to _VAL..."
for src, dst in itertools.izip(files_to_copy, new_file_paths):
	shutil.copy(src, dst)

raw_input("DONE. Delete original files?")
res = raw_input("CONFIRM. Delete original files?<y/n>")
if res=="y":
	for fname in files_to_copy:
		os.remove(fname)

print "Updating DICT_class_TO_images."
indices_to_drop = set(indices_to_drop)
class_to_images_file = "DICT_class_TO_images.pkl"

with open(class_to_images_file, 'r') as fp:
	class_TO_images = pickle.load(fp)

	for c,l in class_TO_images.iteritems():
		class_TO_images[c] = [i for i in l if i not in indices_to_drop]

with open(class_to_images_file, 'w') as fp:
	pickle.dump(class_TO_images, fp)

print "DONE"