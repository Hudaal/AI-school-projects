import numpy as np
import matplotlib.pyplot as plt
from Data_generator import Data_generator

class Viewer:
    def __init__(self, ndim, width, height, noise, image_count_in_set):
        self.width = width
        self.height = height
        self.noise = noise
        self.image_count_in_set = image_count_in_set
        self.ndim = ndim
        self.images = []
        self.classes = []
        self.DG = None

    def generateImages(self):
        self.DG = Data_generator(self.ndim, self.noise, self.height, self.width, image_count_in_set=self.image_count_in_set)
        self.DG.generate_shuffle()
        self.images = self.DG.all_shape_matrix
        self.classes = self.DG.all_shape_class

    def get_image_sets(self):
        return self.DG

    def viewImages(self):
        all_index = 0
        while all_index < len(self.images):
            figure, axis = plt.subplots(4, 4)
            step = 4
            for index in range(step):
                for j in range(step):
                    if all_index >= len(self.images):
                        plt.show()
                        return
                    axis[index, j].imshow(self.images[all_index])
                    if self.classes[all_index][0] == 0.9:
                        axis[index, j].set_title("Circle")
                    elif self.classes[all_index][1] == 0.9:
                        axis[index, j].set_title("Cross")
                    elif self.classes[all_index][2] == 0.9:
                        axis[index, j].set_title("Triangle")
                    elif self.classes[all_index][3] == 0.9:
                        axis[index, j].set_title("Rectangle")
                    else:
                        axis[index, j].set_title("Unknown shape")
                    all_index += 1
        plt.show()

if __name__ == '__main__':
    Viewer = Viewer(ndim=20, width=20, height=20, noise=3, image_count_in_set=500)
    Viewer.generateImages()
    Viewer.viewImages()