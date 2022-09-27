#!/usr/bin/env python3

import os
import shutil
import glob
from functools import reduce
from PIL import Image


RAW_IMAGES_FOLDER = 'raw_images'
EXTRACTED_IMAGES_FOLDER = 'extracted_images'

TRAINING_FOLDERS = {
	'source': 'minecraft_1.16.1',
	'target': 'faithful_1.16.1'
}


shutil.rmtree(EXTRACTED_IMAGES_FOLDER, ignore_errors=True)
os.mkdir(EXTRACTED_IMAGES_FOLDER)


globs = []

for dir in TRAINING_FOLDERS.values():
	RAW_PATH = f'{RAW_IMAGES_FOLDER}/{dir}'

	os.chdir(RAW_PATH)
	globs.append(set(glob.glob('**/*.png', recursive=True)))
	os.chdir('../../')


shared_filenames = reduce(lambda a, b: a & b, globs)

print('found %s images' % len(shared_filenames))

raw_source_img_path = f'{RAW_IMAGES_FOLDER}/{TRAINING_FOLDERS["source"]}'
ext_source_img_path = f'{EXTRACTED_IMAGES_FOLDER}/{TRAINING_FOLDERS["source"]}'

raw_target_img_path = f'{RAW_IMAGES_FOLDER}/{TRAINING_FOLDERS["target"]}'
ext_target_img_path = f'{EXTRACTED_IMAGES_FOLDER}/{TRAINING_FOLDERS["target"]}'

os.mkdir(ext_source_img_path)
os.mkdir(ext_target_img_path)


print(f"copying {dir}")
for filename in shared_filenames:

	source_img = Image.open(f'{raw_source_img_path}/{filename}')
	target_img = Image.open(f'{raw_target_img_path}/{filename}')

	# img2 = Image.open(f'{EXTRACT_PATH}/{filename}')

	if source_img.size != (16, 16):
		continue

	new_filename = filename.replace('/', '.')

	shutil.copy(f'{raw_source_img_path}/{filename}', f'{ext_source_img_path}/{new_filename}')
	shutil.copy(f'{raw_target_img_path}/{filename}', f'{ext_target_img_path}/{new_filename}')
