import urllib.request
import cv2
import numpy as np
import mediapipe as mp
import os

# Download test image
req = urllib.request.Request('https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg', headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as response, open('test.jpg', 'wb') as out_file:
    out_file.write(response.read())

img = cv2.imread('test.jpg')

segmenter_path = os.path.join(os.path.dirname(__file__), "backend", "selfie_multiclass_256x256.tflite")
options = mp.tasks.vision.ImageSegmenterOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=segmenter_path),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    output_category_mask=True
)
segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(options)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

segmentation_result = segmenter.segment(mp_image)
category_mask = segmentation_result.category_mask.numpy_view()

print("Unique classes in mask:", np.unique(category_mask))

hair_mask = (category_mask == 1).astype(np.uint8)
print("Hair mask sum:", np.sum(hair_mask))
