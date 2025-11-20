import numpy as np
from skimage import measure
from scipy.ndimage import binary_erosion, binary_dilation
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import json
from cv2geojson import find_geocontours, export_annotations
def export_geojson(numpy_array, tissue_mask, path):
# suppose your array is called `arr` and contains values in [0,1]
    arr = numpy_array
    structure = np.ones((7,7), dtype=bool)
    eroded = binary_erosion(arr, structure=structure, iterations=10)
    dilated=binary_dilation(eroded, structure=structure, iterations=5)
    eroded=np.asarray(eroded, dtype=np.uint8)
    #cnts, _ = cv2.findContours(dilated.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    tissue_mask = tissue_mask
    seg_cleaned = np.where(tissue_mask, dilated, 0)
    geocontour_opencv = find_geocontours(seg_cleaned, mode='opencv')
    features = [contour.export_feature(color=(0, 255, 0), label='roi') for contour in geocontour_opencv]
    export_annotations(features, path)



