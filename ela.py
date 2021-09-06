import os
import sys
from tqdm import tqdm

from PIL import Image
from PIL import ImageChops
from PIL import ImageEnhance

for filename in tqdm(os.listdir('D:/Code/Tima_Onbroading/ELA/data_4')):
    basename, extension = os.path.splitext(filename)
    resaved = 'img_train_ela/' + basename + '.resaved.jpg'
    ela = 'img_train_ela/' + basename + '.ela.png'
    im = Image.open('D:/Code/Tima_Onbroading/ELA/data_4/' + filename)
    im.save(resaved, 'JPEG', quality=90)
    resaved_im = Image.open(resaved)

    ela_im = ImageChops.difference(im, resaved_im)
    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0/max_diff

    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)

#     print('Maximum difference was {}'.format(max_diff))
    ela_im.save(ela)