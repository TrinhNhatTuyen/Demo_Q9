import cv2
import math
import numpy as np
def padding(imgface,target_size=(224,224)):
    try:
        tile= target_size[0]/(imgface.shape[0]+1)
        he=math.floor(tile*imgface.shape[0])
        wi=math.floor(tile*imgface.shape[1])

        imgface=cv2.resize(imgface,(wi,he))
        target_size=(224, 224)
        factor_0 = target_size[0] / imgface.shape[0]
        factor_1 = target_size[1] / imgface.shape[1]
        factor = min(factor_0, factor_1)
        dsize = (int(imgface.shape[1] * factor), int(imgface.shape[0] * factor))
        imgface = cv2.resize(imgface, dsize)
        # Then pad the other side to the target size by adding black pixels
        diff_0 = target_size[0] - imgface.shape[0]
        diff_1 = target_size[1] - imgface.shape[1]
        # Put the base image in the middle of the padded image
        imgface = np.pad(imgface, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'constant')
        #double check: if target image is not still the same size with target.
        if imgface.shape[0:2] != target_size:
            imgface = cv2.resize(imgface, target_size)
        return imgface
    except:
        print("err padd")
        return imgface
