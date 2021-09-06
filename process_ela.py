#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
from tqdm import tqdm

from PIL import Image
from PIL import ImageChops
from PIL import ImageEnhance


# In[ ]:


for filename in tqdm(os.listdir('D:/Code/Tima_Onbroading/ELA/datatest_private/real')):
    basename, extension = os.path.splitext(filename)
    resaved = 'real/' + basename + '.resaved.jpg'
    ela = 'real/' + basename + '.ela.png'
    im = Image.open('D:/Code/Tima_Onbroading/ELA/datatest_private/real/' + filename)
    im.save(resaved, 'JPEG', quality=90)
    resaved_im = Image.open(resaved)

    ela_im = ImageChops.difference(im, resaved_im)
    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0/max_diff

    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)

#     print('Maximum difference was {}'.format(max_diff))
    ela_im.save(ela)


# In[ ]:


for filename in tqdm(os.listdir('D:/Code/Tima_Onbroading/ELA/fake')):
    if 'ela' not in filename:
        os.remove("D:/Code/Tima_Onbroading/ELA/fake/" + filename)

