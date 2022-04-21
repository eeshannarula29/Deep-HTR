import cv2
import matplotlib.pyplot as plt
import numpy as np


def visualize(img, aabbs, i):
    img = ((img + 0.5) * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for j, aabb in enumerate(aabbs, 0):
        aabb = aabb.enlarge_to_int_grid().as_type(int)
        cv2.rectangle(img, (aabb.xmin, aabb.ymin), (aabb.xmax, aabb.ymax), (255, 0, 255), 2)
        word = img[aabb.ymin:aabb.ymax, aabb.xmin:aabb.xmax]
        cv2.imwrite('../data/test/Cropped/' + str(i) + '_' + str(j) + '.jpg', word)
    return img


def visualize_and_plot(img, aabbs, i):
    for j, aabb in enumerate(aabbs, 0):
        word = img[int(aabb.ymin):int(aabb.ymax), int(aabb.xmin):int(aabb.xmax)]
        plt.axis('off')
        plt.imshow(word, cmap='gray')
        plt.savefig('Word Detector Bkp/data/test/Cropped/' + str(i) + '_' + str(j) + '.jpg')
        plt.close()
    plt.imshow(img, cmap='gray')
    for aabb in aabbs:
        plt.plot([aabb.xmin, aabb.xmin, aabb.xmax, aabb.xmax, aabb.xmin],
                 [aabb.ymin, aabb.ymax, aabb.ymax, aabb.ymin, aabb.ymin])
    plt.show()
