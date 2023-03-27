import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb

class SLIC:
    def __init__(self, image, k, m):
        self.image = image
        self.height, self.width, self.channels = self.image.shape
        self.num_pixels = self.height * self.width
        self.k = k # Number of clusters
        self.m = m # Compactness parameter
        self.cell_size = 0 #regulat grid step
        
        # Initialize cluster
        self.centers = None
        self.distances = np.ones((self.height, self.width)) * np.inf
        self.labels = np.zeros((self.height, self.width), dtype=np.uint16)

    def initialize_clusters(self, image, num_clusters, compactness):
        """
        Randomly initialize centroids
        """
        height, width, channel = image.shape

        # Calculate the cell size and the grid size
        cell_size = np.sqrt(height * width / num_clusters) #average number of pixel in each super-pixel
        grid_size = np.ceil(np.array([height, width]) / cell_size).astype(int)

        # Initialize cluster center
        centers_x = np.linspace(cell_size/2, width-cell_size/2, grid_size[1])
        centers_y = np.linspace(cell_size/2, height-cell_size/2, grid_size[0])
        centers_x, centers_y = np.meshgrid(centers_x, centers_y)
        centers_x = centers_x.reshape(-1)
        centers_y = centers_y.reshape(-1)
        centers = np.zeros((grid_size[0]*grid_size[1], 5))
        centers[:, 0:3] = image[centers_y.astype(int), centers_x.astype(int)]
        centers[:, 3] = centers_y
        centers[:, 4] = centers_x

        # Randomly select the centers
        offset_x = np.random.uniform(-cell_size/2, cell_size/2, centers.shape[0])
        offset_y = np.random.uniform(-cell_size/2, cell_size/2, centers.shape[0])
        centers[:, 3] += offset_y
        centers[:, 4] += offset_x

        # Update the compactness parameter
        self.cell_size = cell_size #cell-size is the scale parameter, s/m?
        #print(cell_size)
        return centers

    def assign_pixels_to_clusters(self):
        """
        Assign pixels to each cluster by calculating distance based on color
        """
        for i in range(self.k):
            # Define search window
            # The paper proposed search box 2S, but not working well. This one (m*S) seems work the best.
            x_min = int(max(self.centers[i, 4] - self.cell_size*self.m, 0))
            x_max = int(min(self.centers[i, 4] + self.cell_size*self.m, self.width))
            y_min = int(max(self.centers[i, 3] - self.cell_size*self.m, 0))
            y_max = int(min(self.centers[i, 3] + self.cell_size*self.m, self.height))

            window = self.image[y_min:y_max, x_min:x_max]

            window_lab = rgb2lab(window) #use CEILAB color

            # Calculate distances
            distances = np.sqrt(np.sum((window_lab - self.centers[i, :3])**2, axis=2))
            y, x = np.indices(distances.shape)
            distances += np.sqrt((x - self.centers[i, 4])**2 + (y - self.centers[i, 3])**2) / (self.m*self.cell_size) #paper suggest s/m, but this one seems have better performance on my image

            # Update labels and distances
            mask = distances < self.distances[y_min:y_max, x_min:x_max]
            self.labels[y_min:y_max, x_min:x_max][mask] = i
            self.distances[y_min:y_max, x_min:x_max][mask] = distances[mask]
        
    def update_cluster_centers(self):
        """
        Update the center of each cluster
        """
        for i in range(self.k):
            mask = self.labels == i
            if np.any(mask):
                self.centers[i, :3] = np.mean(self.image[mask], axis=0)
                y, x = np.indices(self.labels.shape)
                self.centers[i, 4] = np.mean(x[mask])
                self.centers[i, 3] = np.mean(y[mask])
        
    def generate_superpixels(self, num_iterations=10):
        """
        Generate the Superpixel
        """
        self.centers = self.initialize_clusters(self.image,self.k,self.m)
        for i in range(num_iterations):
            print(f"Iteration {i + 1}")
            self.assign_pixels_to_clusters()
            self.update_cluster_centers()
            self.labels = self.labels.astype(int)
            print(f"Max label: {np.max(self.labels)}")


    def visualize_superpixels(self):
        """
        Visualize the result
        """
        labels_rgb = lab2rgb(self.centers[:, :3])
        result = labels_rgb[self.labels.ravel()].reshape(self.image.shape) #Map each pixel
        result = result * 255 #convert to int
        Image.fromarray(np.uint8(result)).show()

    