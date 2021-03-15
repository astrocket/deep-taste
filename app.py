import os
import numpy
import PIL
from PIL import ImageFilter
from PIL import Image
import keras
from keras.preprocessing.image import img_to_array
from keras.models import load_model

model = load_model('./font_finder.h5')

directory = os.path.join(os.path.dirname(__file__), "test_set")

labelToIndex = {}
indexToLabel = {}

fontDirectories = os.path.join(os.path.dirname(__file__), "../noonnu")

for _i, fontDirectory in enumerate(os.listdir(fontDirectories)):
  fontDirectoryName = os.fsdecode(fontDirectory)
  labelToIndex[fontDirectory] = _i
  indexToLabel[_i] = fontDirectory

for file in os.listdir(directory):
  filename = os.fsdecode(file)
  img_path = os.path.join(directory, filename)
  pil_im=Image.open(img_path).convert('L')
  pil_im=pil_im.resize((105, 105))
  org_img=img_to_array(pil_im)

  data=[]
  data.append(org_img)
  data = numpy.asarray(data, dtype="float") / 255.0

  y = model.predict_classes(data)
  answer = indexToLabel[int(y[0])]
  print(img_path)
  print('%s ===> %s'%(filename,answer))