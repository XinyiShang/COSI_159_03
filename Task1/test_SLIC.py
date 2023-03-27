import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries

from self_SLIC import SLIC

#change to load different images
img = np.array(Image.open('brandeis_2.jpg'))

#My implementation of SLIC
slic_impl = SLIC(img, 20, 10)
slic_impl.generate_superpixels()
segments_impl = slic_impl.labels

#scikit-image
package_slic = slic(img, n_segments=20, compactness=10)

# Display the results
fig, ax = plt.subplots(1, 2)

ax[0].imshow(mark_boundaries(img, package_slic))
ax[0].set_title('SLIC with scikit-image')

ax[1].imshow(mark_boundaries(img, segments_impl))
ax[1].set_title('My implementation of SLIC')

plt.show()
