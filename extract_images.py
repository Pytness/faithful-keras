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



def transform_alpha_image(img: Image):
	# Set all transparent pixels to black

	# This is done to make the model more robust to transparent blocks
	# Get the alpha band
	red, green, blue, alpha = img.split()
	
	for i, pixel in enumerate(alpha.getdata()):
		x = i % img.width
		y = i // img.width
		if pixel == 0:
			red.putpixel((x, y), 0)
			green.putpixel((x, y), 0)
			blue.putpixel((x, y), 0)

def generate_deformations(img: Image):
	"""Generate all 8 possible deformations of an image"""

	# 1. Rotate 90 degrees
	# 2. Rotate 180 degrees
	# 3. Rotate 270 degrees
	# 4. Flip horizontally
	# 5. Flip vertically
	# 6. Flip horizontally and rotate 90 degrees
	# 7. Flip horizontally and rotate 270 degrees
	# 8. Flip vertically and rotate 90 degrees
	# 9. Flip vertically and rotate 270 degrees

	# 1. Rotate 90 degrees
	yield img.rotate(90)

	# 2. Rotate 180 degrees
	yield img.rotate(180)

	# 3. Rotate 270 degrees
	yield img.rotate(270)

	# 4. Flip horizontally
	yield img.transpose(Image.FLIP_LEFT_RIGHT)

	# 5. Flip vertically
	yield img.transpose(Image.FLIP_TOP_BOTTOM)

	# 6. Flip horizontally and rotate 90 degrees
	yield img.transpose(Image.FLIP_LEFT_RIGHT).rotate(90)

	# 7. Flip horizontally and rotate 270 degrees
	yield img.transpose(Image.FLIP_LEFT_RIGHT).rotate(270)

	# 8. Flip vertically and rotate 90 degrees
	yield img.transpose(Image.FLIP_TOP_BOTTOM).rotate(90)

	# 9. Flip vertically and rotate 270 degrees
	yield img.transpose(Image.FLIP_TOP_BOTTOM).rotate(270)

	yield img


print(f"copying {dir}")
for filename in shared_filenames:

	source_img = Image.open(f'{raw_source_img_path}/{filename}').convert('RGBA')
	target_img = Image.open(f'{raw_target_img_path}/{filename}').convert('RGBA')

	print(filename, source_img.mode)
	transform_alpha_image(source_img)
	print(filename, target_img.mode)
	transform_alpha_image(target_img)

	# img2 = Image.open(f'{EXTRACT_PATH}/{filename}')

	if source_img.size != (16, 16):
		continue

	new_filename = filename.replace('/', '.')

	for i, img in enumerate(generate_deformations(source_img)):
		img.save(f'{ext_source_img_path}/{i}_{new_filename}.png')

	for i, img in enumerate(generate_deformations(target_img)):
		img.save(f'{ext_target_img_path}/{i}_{new_filename}.png')



	# shutil.copy(f'{raw_source_img_path}/{filename}', f'{ext_source_img_path}/{new_filename}')
	# shutil.copy(f'{raw_target_img_path}/{filename}', f'{ext_target_img_path}/{new_filename}')
