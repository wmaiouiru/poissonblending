#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import scipy.sparse
import PIL.Image
import pyamg
import rasterio
import argparse
def main(argv):
    argument_sample='python '+os.path.realpath(__file__)
    parser = argparse.ArgumentParser(description=argument_sample)
    parser.add_argument('-src_file',dest='src_file',nargs='?',help='src_file',default=None)
    parser.add_argument('-ref_file',dest='ref_file',nargs='?',help='ref_file',default=None)
    parser.add_argument('-output_file',dest='output_file',nargs='?',help='output_file',default=None)
    parser.add_argument('-mask_file',dest='mask_file',nargs='?',help='mask_file',default=None)

    args = parser.parse_args()

    srcpath=args.src_file
    refpath=args.ref_file
    dstpath=args.output_file
    mask_file=args.mask_file
    if srcpath and refpath and dstpath:
        with rasterio.open(srcpath) as src:
            r = src.read(1)

            mask = np.clip(r, 0, 1)
        img_source = np.asarray(PIL.Image.open(srcpath))
        img_source.flags.writeable = True
        img_target = np.asarray(PIL.Image.open(refpath))
        img_target.flags.writeable = True
        img_ret = blend(img_target, img_source, mask)
        img_ret = PIL.Image.fromarray(np.uint8(img_ret))
        img_ret.save(dstpath)
    else:
        print parser.print_help()
        return 0
# pre-process the mask array so that uint64 types from opencv.imread can be adapted
def prepare_mask(mask):
    if type(mask[0][0]) is np.ndarray:
        result = np.ndarray((mask.shape[0], mask.shape[1]), dtype=np.uint16)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if sum(mask[i][j]) > 0:
                    result[i][j] = 1
                else:
                    result[i][j] = 0
        mask = result
    return mask

def blend(img_target, img_source, img_mask, offset=(0, 0)):
    # compute regions to be blended
    region_source = (
            max(-offset[0], 0),
            max(-offset[1], 0),
            min(img_target.shape[0]-offset[0], img_source.shape[0]),
            min(img_target.shape[1]-offset[1], img_source.shape[1]))
    region_target = (
            max(offset[0], 0),
            max(offset[1], 0),
            min(img_target.shape[0], img_source.shape[0]+offset[0]),
            min(img_target.shape[1], img_source.shape[1]+offset[1]))
    region_size = (region_source[2]-region_source[0], region_source[3]-region_source[1])

    # clip and normalize mask image
    img_mask = img_mask[region_source[0]:region_source[2], region_source[1]:region_source[3]]
    img_mask = prepare_mask(img_mask)
    img_mask[img_mask==0] = False
    img_mask[img_mask!=False] = True
    print 'mask prepared '

    # create coefficient matrix
    A = scipy.sparse.identity(np.prod(region_size), format='lil')
    for y in range(region_size[0]):
        for x in range(region_size[1]):
            if img_mask[y,x]:
                index = x+y*region_size[1]
                A[index, index] = 4
                if index+1 < np.prod(region_size):
                    A[index, index+1] = -1
                if index-1 >= 0:
                    A[index, index-1] = -1
                if index+region_size[1] < np.prod(region_size):
                    A[index, index+region_size[1]] = -1
                if index-region_size[1] >= 0:
                    A[index, index-region_size[1]] = -1
    A = A.tocsr()
    print 'created coefficient matrix'
    # create poisson matrix for b
    P = pyamg.gallery.poisson(img_mask.shape)

    # for each layer (ex. RGB)
    for num_layer in range(img_target.shape[2]):
        # get subimages
        t = img_target[region_target[0]:region_target[2],region_target[1]:region_target[3],num_layer]
        s = img_source[region_source[0]:region_source[2], region_source[1]:region_source[3],num_layer]
        t = t.flatten()
        s = s.flatten()

        # create b
        b = P * s
        for y in range(region_size[0]):
            for x in range(region_size[1]):
                if not img_mask[y,x]:
                    index = x+y*region_size[1]
                    b[index] = t[index]

        # solve Ax = b
        x = pyamg.solve(A,b,verb=False,tol=1e-10)

        # assign x to target image
        x = np.reshape(x, region_size)
        x[x>255] = 255
        x[x<0] = 0
        x = np.array(x, img_target.dtype)
        img_target[region_target[0]:region_target[2],region_target[1]:region_target[3],num_layer] = x

    return img_target


def test():
    img_mask = np.asarray(PIL.Image.open('./testimages/test1_mask.png'))
    img_mask.flags.writeable = True
    img_source = np.asarray(PIL.Image.open('./testimages/test1_src.png'))
    img_source.flags.writeable = True
    img_target = np.asarray(PIL.Image.open('./testimages/test1_target.png'))
    img_target.flags.writeable = True
    img_ret = blend(img_target, img_source, img_mask, offset=(40,-30))
    img_ret = PIL.Image.fromarray(np.uint8(img_ret))
    img_ret.save('./testimages/test1_ret.png')


if __name__ == '__main__':
    main(sys.argv[1:])
