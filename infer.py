import argparse

from path import Path
import numpy as np
import os

from dataloader import DataLoaderImgFile
from eval import evaluate
from net import WordDetectorNet
from visualization import visualize_and_plot
from PIL import Image
from model import *


def qsort(inlist):
    if inlist == []:
        return []
    else:
        pivot = inlist[0].ymin
        lesser = qsort([y for y in inlist[1]])

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
#     args = parser.parse_args()
#
#     net = WordDetectorNet()
#     net.load_state_dict(torch.load('../model/weights', map_location=args.device))
#     net.eval()
#     net.to(args.device)
#
#     loader = DataLoaderImgFile(Path('../data/test'), net.input_size, args.device)
#     res = evaluate(net, loader, max_aabbs=1000)
#
#     for i, (img, aabbs) in enumerate(zip(res.batch_imgs, res.batch_aabbs)):
#         f = loader.get_scale_factor(i)
#         aabbs = [aabb.scale(1 / f, 1 / f) for aabb in aabbs]
#         number = int(np.floor(np.log10(len(aabbs))))
#         ymins = {}
#         for j, aabb in enumerate(aabbs, 0):
#             ymins[j] = aabb.ymin
#         ymins = sorted(ymins, key=ymins.get)
#         aabbs_new = []
#         for ymin in ymins:
#             aabbs_new.append(aabbs[ymin])
#         start_line = 0
#         for j in range(1, len(aabbs_new)):
#             if aabbs_new[j].ymin > aabbs_new[j-1].ymax or j == len(aabbs_new)-1:
#                 if j == len(aabbs_new)-1:
#                     j += 1
#                 xmins = {}
#                 for k, aabb in enumerate(aabbs[start_line:j], start_line):
#                     xmins[k] = aabb.xmin
#                 xmins = sorted(xmins, key=xmins.get)
#                 aabbs_temp = []
#                 for xmin in xmins:
#                     aabbs_temp.append(aabbs[xmin])
#                 aabbs_new[start_line:j] = aabbs_temp
#                 start_line = j
#         img = loader.get_original_img(i)
#         visualize_and_plot(img, aabbs_new, i)
#
#     words = '../data/test/Cropped'
#     model = Net()
#     model.load_state_dict(torch.load('../model/classifier.pt'))
#     paths = []
#     for image in sorted(os.listdir(words)):
#         if image == '.DS_Store':
#             continue
#         f = os.path.join(words, image)
#         paths.append(f)
#     paths.sort(key=os.path.getctime)
#
#     for path in paths:
#         read_word_trocr(path)
#
#
# def read_word_trocr(path):
#     print(path)
#     model_path = '/trocr-base-handwritten.pt'
#     jpg_path = path
#     beam = 5
#
#     model, cfg, task, generator, bpe, img_transform, device = init(model_path, beam)
#
#     sample = preprocess(jpg_path, img_transform)
#
#     text = get_text(cfg, generator, model, sample, bpe)
#
#     print(text)


if __name__ == '__main__':
    # main()
    pass
