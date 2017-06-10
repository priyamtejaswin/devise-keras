import os
import re
import sys

file_path = sys.argv[1]
image_re = re.compile('<td><img src="(.*)\/(.*)"><\/td>')
caption_re = re.compile('<tr><td>(.*)<\/td><\/tr>')

UIUC_ROOT = "UIUC_PASCAL_DATA"
UIUC_URL = "http://vision.cs.uiuc.edu/pascal-sentences"

if os.path.exists(UIUC_ROOT):
	raw_input("\nUIUC_PASCAL_DATA detected. The program has stopped here. Press ENTER to continue downloading all data. Press CTRL+C to exit program now.\n")
else:
	os.makedirs(UIUC_ROOT)

print "\nParsing and downloading html source...\n"

_c = 0
with open(file_path, 'r') as fp:
	for line in fp.readlines():
		_c+=1
		if _c%50==0:
			print "Completed", _c

		clean = line
		print clean

		match_image = image_re.search(clean)
		if match_image:
			class_name = match_image.group(1)
			image_name = match_image.group(2)

			dir_name = os.path.join(UIUC_ROOT, class_name)
			if not os.path.exists(dir_name):
				os.makedirs(dir_name)

			img_name = os.path.join(dir_name, image_name)
			img_url = os.path.join(UIUC_URL, class_name, image_name)
			system_string = "wget %s -O %s"%(img_url, img_name)
			os.system(system_string)
			continue

		match_caption = caption_re.search(clean)
		if match_caption:
			caption_text = match_caption.group(1).strip()
			print caption_text

		raw_input("PAUSE")