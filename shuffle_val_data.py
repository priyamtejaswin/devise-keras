import os, sys, shutil, itertools
import ConfigParser

config = ConfigParser.RawConfigParser()
config.read('local.cfg')
TRAINING_DATA_FOLDER 	= config.get("data location", "TRAINING_DATA_FOLDER")
VALIDATION_DATA_FOLDER  = config.get("data location", "VALIDATION_DATA_FOLDER")
print "TRAINING_DATA_FOLDER: ", TRAINING_DATA_FOLDER, " VALIDATION_DATA_FOLDER: ", VALIDATION_DATA_FOLDER

count = 1

files_to_copy, new_file_paths = [], []

for root, dirs, files in os.walk(TRAINING_DATA_FOLDER):
	for fname in files:
		if "DS_Store" in fname:
			continue
		
		if count%5==0:
			files_to_copy.append(os.path.join(root, fname))
			print root, fname

			new_root = root.replace("_DATA", "_VAL")
			# new_root   = VALIDATION_DATA_FOLDER
			if not os.path.isdir(new_root):
				os.mkdir(new_root)

			new_file_paths.append(os.path.join(new_root, fname))

		count+=1

print "Beginning to copy %d files from _DATA to _VAL..."
for src, dst in itertools.izip(files_to_copy, new_file_paths):
	shutil.copy(src, dst)

raw_input("DONE. Delete original files?")
res = raw_input("CONFIRM. Delete original files?<y/n>")
if res=="y":
	for fname in files_to_copy:
		os.remove(fname)