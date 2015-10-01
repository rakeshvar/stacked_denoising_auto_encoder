#!/usr/bin/python

import os
import sys
import numpy
from code_folder.utils import tile_raster_images
import PIL.Image
from sklearn.decomposition import PCA

def read_json_bz2(path2data):
    import bz2,json,contextlib
    with contextlib.closing(bz2.BZ2File(path2data, 'rb')) as f:
        return numpy.array(json.load(f))

def get_pcs(dataset='numbers.x.bz2',
            output_folder='dA_plots',
            n_comps = (10, 10)):
    print "Reading Data"
    x_data = read_json_bz2(dataset)
    print x_data.shape    

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)
    
    n_features = x_data.shape[1]
    side = int(n_features ** .5)
    assert(side ** 2 == n_features)      # Need perfect square

    image2 = PIL.Image.fromarray(tile_raster_images(
                            X=x_data, scale_rows_to_unit_interval=False,
                            img_shape=(side, side), tile_shape=n_comps,
                            tile_spacing=(1, 1)))

    image2.save('sample_x.png')

    print "PCAing"
    pca = PCA(n_components=n_comps[0]*n_comps[1])
    pca.fit(x_data)
    print(pca.explained_variance_ratio_) 

    print "Saving images"

    image = PIL.Image.fromarray(tile_raster_images(
                            X=pca.components_,
                            img_shape=(side, side), tile_shape=n_comps,
                            tile_spacing=(1, 1)))

    image.save(dataset+'pca.png')


    os.chdir('../')
    print "Raster saved"


if __name__ == '__main__':
    kwargs = dict([arg.split('=', 1) for arg in sys.argv[1:]])
    get_pcs(**kwargs)
