# Segment via semantic segmentation
import pixellib
from pixellib.semantic import semantic_segmentation
from PIL import Image
import numpy as np
import cv2
import sys

def segmentation_attack_mask(segment_image, img_path):
    kosher_classes = [1, 2, 4, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 19, 20, 21, 23, 24, 25, 26, 29, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 70, 71, 72, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 87, 89, 90, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 142, 143, 144, 145, 146, 147, 149]

    img = cv2.imread(img_path)
    segvalues, object_masks, output = segment_image.segmentFrameAsAde20k(img, extract_segmented_objects=True)
    masks_final = np.zeros((np.shape(img)[0], np.shape(img)[1]))
    # Get the final mask
    for mask in object_masks:
        if mask['class_id'] in kosher_classes:
            masks_final = np.logical_or(masks_final, np.array(Image.fromarray(mask['masks'].astype('uint8')).resize((img.shape[1], img.shape[0]))))
    del segvalues
    del object_masks
    del output
    az_util = np.sum(masks_final) / (np.shape(masks_final)[0] * np.shape(masks_final)[1])
    return masks_final, az_util
#        resize_masks = np.array(Image.fromarray(segvalues["masks"].astype('uint8')).resize((h, w)))

#print("Final shape: ")
#print(np.shape(masks_final))
#print("Malaka: ")
#print(masks_final)
#final_image = np.multiply(cv2.imread("0001.jpg"), np.stack((masks_final,)*3, axis=-1))
#cv2.imwrite("out.jpg", final_image)
##np.savetxt("masks.txt", segvalues["masks"])
#print(segvalues["class_ids"])
#
