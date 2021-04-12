#import torch
#print(torch.cuda.is_available())

import os
import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt
i = 0 # Increment variable


for filename in os.listdir('training_data/new_data/'):
    filepath = os.path.join('training_data/new_data/', filename)
    #fd = open(filepath, 'r')

    array = np.load(filepath)
    img = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    i += 1
    #cv2.imwrite('training_data/npy2jpg/'+str(i)+'.jpg', img)

    print('File: ' + str(filepath) + '\n' + 'Converted to: ' + 'training_data/npy2jpg/' + str(i) + '.jpg')

    cv2.waitKey(0)