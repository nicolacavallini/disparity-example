#from __future__ import division
#import scipy as scp
from skimage import data, io, filters, transform
import numpy as np
import matplotlib.pyplot as plt

from sys import stdout



def read_image(lr,ref_id):
    img_name = "./stereo_imgs/rectified_" + lr + "_" + ref_id +".png"
    img = io.imread(img_name,as_grey=True)
    #img[np.where(img==0)]=1
    return img

def noralize_patch(pl):
    pl = pl - np.mean(pl)
    pl = pl/np.amax(pl.flatten())
    ids = np.isnan(pl)
    pl[ids] = 1.
    return pl

def get_patch_list(left,semi_stencil):
    patch_l = []
    for i in range(left.shape[0]):
        row = []
        for j in range(left.shape[1]):
            row.append(np.zeros((2*semi_stencil+1,2*semi_stencil+1)))
        patch_l.append(row)


    for i in range(semi_stencil,left.shape[0]-semi_stencil):
        for j in range(semi_stencil,left.shape[1]-semi_stencil):
            p = left[i-semi_stencil:i+semi_stencil,
                                j-semi_stencil:j+semi_stencil]
            #p = noralize_patch(p)
            patch_l[i][j] = p
    return patch_l

def get_roi(left):
    rows = np.where(np.sum(left,1)>0)[0]
    cols = np.where(np.sum(left,0)>0)[0]
    return rows,cols#left[rows[0]:rows[-1],cols[0]:cols[-1]]

def mask_sobel(sobel):
    mask = np.zeros(sobel.shape,dtype=np.int32)
    ids = np.where(sobel>.007)
    mask[ids] = 1
    return mask

def from_mask_to_pixels_ids(mask):
    ids = np.where(mask==1)

    return pixels

def evaluate_matches(mask_left,edge_points,
                     left_patches,right_patches):
    match_points = np.zeros(mask_left.shape,dtype=np.int32)

    for i,j in zip(edge_points[0],edge_points[1]):

        cost = 255*np.ones(mask_left.shape[1])
        pl = left_patches[i][j]
        #pl = noralize_patch(pl)
        for c in right_cols:
            pr = right_patches[i][c]
            k = np.linalg.norm(pl-pr)
            cost[c] = k

        match_points[i,np.argmin(cost)] = 1
        stdout.write("\r evaluating disparity... %d, of %d" % (i,np.amax(edge_points[0])))
        stdout.flush()

    stdout.write("\n")

    return match_points


if __name__ == "__main__":
    left = read_image("left","03")
    right = read_image("right","03")

    #left = get_roi(left)
    left_rows,left_cols = get_roi(left)
    right_rows,right_cols = get_roi(right)

    left_sobel = filters.sobel(left)
    right_sobel = filters.sobel(right)

    mask_left = mask_sobel(left_sobel)
    mask_right = mask_sobel(right_sobel)

    edge_points = np.where(mask_left==1)

    print edge_points[0].shape

    semi_stencil = 5
    #
    left_patches = get_patch_list(left,semi_stencil)
    right_patches = get_patch_list(right,semi_stencil)


    for i,j in zip(edge_points[0],edge_points[1]):
        left_patches[i][j] = noralize_patch(left_patches[i][j])
        stdout.write("\r left normalizing... %d, of %d" % (i,np.amax(edge_points[0])))
        stdout.flush()
    stdout.write("\n")

    for i in right_rows:
        for j in right_cols:
            right_patches[i][j] = noralize_patch(right_patches[i][j])
        stdout.write("\r right normalizing... %d, of %d" % (i,np.amax(right_rows)))
        stdout.flush()
    stdout.write("\n")

    match_points = evaluate_matches(mask_left,edge_points,
                         left_patches,right_patches)



    #print i,j

    plt.subplot(2,2,1)
    plt.imshow(mask_left,cmap='winter')
    plt.subplot(2,2,2)
    plt.imshow(mask_right,cmap='winter')
    plt.subplot(2,2,3)
    plt.imshow(mask_left,cmap='winter')
    plt.subplot(2,2,4)
    plt.imshow(match_points,cmap='cool')
    plt.show()

    #pixels =  from_mask_to_pixels_ids(mask_left)


    # print left.shape
    #
    # right = get_roi(right)
    # right = transform.resize(right,left.shape)
    # print right.shape

    #left = left[rows,cols]

    #plt.plot(mrk)
    #plt.show()

    # disparity = np.zeros(left.shape)

    # print len(left_patches), len(left_patches[0])
    # print len(right_patches), len(right_patches[0])
    #

    #

    # disparity = np.zeros(left.shape)
    #


        # stdout.write("\r evaluating disparity ... %d, of %d" % (i,np.amax(rows)))
        # stdout.flush()
    # plt.imshow(disparity)
    # plt.show()



    # plt.subplot(2,2,1)
    # plt.imshow(left_patches[224][250])
    # plt.subplot(2,2,2)
    # plt.imshow(right_patches[224][96])
    #
    # plt.subplot(2,2,3)
    # plt.imshow(left)
    # plt.subplot(2,2,4)
    # plt.imshow(right)
    # plt.show()
