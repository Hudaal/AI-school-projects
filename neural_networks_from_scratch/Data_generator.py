import numpy as np
import matplotlib.pyplot as plt
import random

class Data_generator:
    def __init__(self, dim=10, noise=10, image_height=10, image_width=10, image_count_in_set=100):
        self.dim = dim
        self.noise = noise
        self.image_height = image_height
        self.image_width = image_width
        self.image_count_in_set = image_count_in_set
        self.all_shape_matrix = [np.zeros((self.image_height, self.image_width)) for _ in range(image_count_in_set)]
        self.all_shape_class = [np.zeros(4) for _ in range(image_count_in_set)]
        self.train_shapes = []
        self.validate_shapes = []
        self.test_shapes = []

        self.ccc = 0

    def _2d_matrix_to_vectors(self, shape_matrix):
        shape_matrix_vectors = []
        for row in shape_matrix:
            for value in row:
                shape_matrix_vectors.append(value)
        return shape_matrix_vectors

    def fletting_into_vectors(self, shape_class_list, size):
        if size <= 0:
            return False
        fletted_list_vectors = []
        fletted_list_classes = []
        for i in range(0, len(shape_class_list), size):
            one_shape_vector = []
            one_shape_class = []
            for e in range(size):
                for vector in self._2d_matrix_to_vectors(shape_class_list[i + e][0]):
                    one_shape_vector.append(vector)
                for j in range(4):
                    one_shape_class.append(shape_class_list[i + e][1][j])
            fletted_list_vectors.append(one_shape_vector)
            fletted_list_classes.append(one_shape_class)
        return fletted_list_vectors, fletted_list_classes

    def circle_y(self, x, radius, center_x, center_y):
        # The circle equation is (x-a)^2 + (y-b)^2 = r^2
        sqrt_result = round(np.sqrt(np.power(radius, 2) - np.power(x - center_x, 2)))
        return center_y + sqrt_result, center_y - sqrt_result

    def cross_y(self, x, a, b):
        return x + a - b, (-1*x) + a + b

    def create_one_circle(self, center, index, radius=10):
        x1_old, x2_old = self.circle_y(center[0] - radius, radius, center[0], center[1])
        noise = self.noise
        i = 0
        noise_indexes = random.choices(range(center[0] - radius, center[0] + radius + 1), k=noise)
        for y in range(center[0] - radius, center[0] + radius + 1):
            x1, x2 = self.circle_y(y, radius, center[0], center[1])
            self.all_shape_matrix[index][y][x1] = 1
            self.all_shape_matrix[index][y][x2] = 1
            if x1 == x2:
                self.all_shape_matrix[index][y][x1 + 1] = 1
                self.all_shape_matrix[index][y][x1 - 1] = 1
            if abs(x1 - x1_old) > 1:
                for i in range(1, abs(x1 - x1_old)):
                    self.all_shape_matrix[index][y][x1 + 1] = 1
                    self.all_shape_matrix[index][y][x1 - i] = 1
            if abs(x2 - x2_old) > 1:
                for i in range(abs(x2 - x2_old)):
                    self.all_shape_matrix[index][y][x2 - 1] = 1
                    self.all_shape_matrix[index][y][x2 + i] = 1
            if y in noise_indexes and i < noise and noise > 0:
                self.all_shape_matrix[index][y][x1] = 0
                replace_index = np.random.randint(center[0] - radius, center[0] + radius, 2)
                self.all_shape_matrix[index][replace_index[0]][replace_index[1]] = 1
                noise -= 1
                i += 1
            x1_old = x1
            x2_old = x2

    def create_circles(self, double=False):
        count = int(self.image_count_in_set / 4)
        index = 0
        while count > 0:
            start = int(self.image_width / 10)
            step = start
            for y in range(start-1, self.image_height, step):
                for x in range(start-1, self.image_width, step):
                    center = [int(y/2), int(x/2)]
                    for r in range(start, int((min(x, y)/2)), step):
                        self.create_one_circle(center, index, radius=r)
                        self.all_shape_class[index] = np.array([0.9, 0.1, 0.1, 0.1])
                        self.ccc += 1
                        if double:
                            self.create_one_circle(center, index, radius=r+1)
                        count -= 1
                        index += 1
                        if count == 0:
                            return
            if not double: double = True
            else: double = False

    def create_one_cross(self, start, end, v1, v2, index, double):
        noise_indexes = np.random.randint(start, end, size=self.noise)
        noise = self.noise
        for x in range(start, end):
            y1, y2 = self.cross_y(x, v1, v2)
            if 0 < y1 < self.image_height-1:
                self.all_shape_matrix[index][y1][x] = 1
                if double:
                    self.all_shape_matrix[index][y1+1][x] = 1
                if x in noise_indexes and noise > 0:
                    self.all_shape_matrix[index][y1][x] = 0
                    self.all_shape_matrix[index][y1][random.randint(start, end)] = 1
                    noise -= 1
            if 0 < y2 < self.image_width-1:
                self.all_shape_matrix[index][x][y2] = 1
                if double:
                    self.all_shape_matrix[index][x + 1][y2] = 1
                if x in noise_indexes and noise > 0:
                    self.all_shape_matrix[index][x][y2] = 0
                    self.all_shape_matrix[index][random.randint(start, end)][y2] = 1
                    noise -= 1

    def create_crosses(self, double=False):
        count = int(self.image_count_in_set / 4)
        index = int(self.image_count_in_set / 4)
        while count > 0:
            step = int(min(self.image_width, self.image_height) / 10)
            for v1 in range(int(self.image_width/2)-1, 1, -step+1):
                for v2 in range(int(self.image_height/2)-2, 1, -step+1):
                    if v1 != v2:
                        self.create_one_cross(step, min(self.image_width, self.image_height) - step, v2, v1, index, double)
                        self.all_shape_class[index] = np.array([0.1, 0.9, 0.1, 0.1])
                        self.ccc += 1
                        count -= 1
                        index += 1
                        if count == 0:
                            return
            if not double: double = True
            else: double = False

    def create_one_triangle(self, side_length, start_x, start_y, index, double):
        noise_indexes = np.random.randint(start_x, start_x + side_length, size=self.noise)
        noise = self.noise
        for i in range(side_length):
            self.all_shape_matrix[index][start_y + i][start_x - i] = 1
            self.all_shape_matrix[index][start_y + i][start_x + i] = 1
            self.all_shape_matrix[index][start_y + side_length][start_x - i] = 1
            self.all_shape_matrix[index][start_y + side_length][start_x + i] = 1

            if double:
                self.all_shape_matrix[index][start_y + i][start_x - i - 1] = 1
                self.all_shape_matrix[index][start_y + i][start_x + i + 1] = 1
                self.all_shape_matrix[index][start_y + side_length][start_x - i - 1] = 1
                self.all_shape_matrix[index][start_y + side_length][start_x + i + 1] = 1

            if start_x + i in noise_indexes and noise > 0:
                self.all_shape_matrix[index][start_y + i][start_x + i] = 0
                self.all_shape_matrix[index][start_y + i][random.randint(1, self.image_width-1)] = 1
                noise -= 1
            if start_x - i in noise_indexes and noise > 0:
                self.all_shape_matrix[index][start_y + i][start_x - i] = 0
                self.all_shape_matrix[index][start_y + i][random.randint(1, self.image_width-1)] = 1
                noise -= 1

    def create_triangles(self, double=False):
        count = int(self.image_count_in_set / 4)
        index = int((self.image_count_in_set / 4) * 2)
        while count > 0:
            step = int(min(self.image_width, self.image_height)/10)
            for length in range(step+1, int(self.image_width/2)-1):
                for x in range(2, int(self.image_width/2), step):
                    for y in range(2, int(self.image_height/2), step):
                        if length <= x:
                            self.create_one_triangle(length, x, y, index, double)
                            self.all_shape_class[index] = np.array([0.1, 0.1, 0.9, 0.1])
                            self.ccc += 1
                            index += 1
                            count -= 1
                            if count == 0:
                                return
            if not double: double = True
            else: double = False

    def create_one_rectangle(self, start_x, start_y, w_length, h_length, index):
        noise_indexes = np.random.randint(start_y, start_y + h_length + 1, size=self.noise)
        noise = self.noise
        for i in range(w_length):
            if i in noise_indexes and noise > 0:
                self.all_shape_matrix[index][random.randint(1, self.image_height-1)][random.randint(1, self.image_height-1)] = 1
                noise -= 1
            else:
                self.all_shape_matrix[index][start_y][start_x + i] = 1
            self.all_shape_matrix[index][start_y + h_length][start_x + i] = 1
        for i in range(h_length + 1):
            if i in noise_indexes and noise > 0:
                self.all_shape_matrix[index][random.randint(1, self.image_height-1)][random.randint(1, self.image_height-1)] = 1
                self.all_shape_matrix[index][start_y + i][random.randint(1, self.image_height-1)] = 1
                noise -= 2
            else:
                self.all_shape_matrix[index][start_y + i][start_x + w_length] = 1
                self.all_shape_matrix[index][start_y + i][start_x] = 1
        if noise > 0:
            for n in range(noise):
                self.all_shape_matrix[index][start_x + n][start_y + n] = 0
                self.all_shape_matrix[index][random.randint(1, self.image_height - 1)][
                    random.randint(1, self.image_height - 1)] = 1

    def create_rectangles(self):
        step = int(min(self.image_width, self.image_height) / 10)
        index = int((self.image_count_in_set / 4) * 3)
        count = int(self.image_count_in_set / 4)
        while count > 0:
            for x_length in range(step, self.image_width, step):
                for y_length in range(step, self.image_height, step):
                    for start in range(1, self.image_width-max(x_length, y_length)-1, step):
                        self.create_one_rectangle(start, start, x_length, y_length, index)
                        self.all_shape_class[index] = np.array([0.1, 0.1, 0.1, 0.9])
                        self.ccc += 1
                        index += 1
                        count -= 1
                        if count == 0:
                            return

    def generate_shuffle(self):
        """This function creates the images ordered, and then shuffle them to send them to the network"""
        self.create_circles()
        self.create_crosses()
        self.create_triangles()
        self.create_rectangles()

        shuffler = np.random.permutation(len(self.all_shape_matrix))
        all_shape_matrix_shuffled = []
        all_shape_classes_shuffled = []
        for i in shuffler:
            all_shape_matrix_shuffled.append(self.all_shape_matrix[i])
            all_shape_classes_shuffled.append(self.all_shape_class[i])
        self.all_shape_matrix = all_shape_matrix_shuffled
        self.all_shape_class = all_shape_classes_shuffled

    def generate_split_image_sets(self, train_percentage, valid_percentage, test_percentage):
        """This function split the images into train, test and validation data sets"""
        train_set_count = int(self.image_count_in_set * train_percentage / 100)
        validation_set_count = int(self.image_count_in_set * valid_percentage / 100)
        test_set_count = int(self.image_count_in_set * test_percentage / 100)

        self.generate_shuffle()
        added_count = 0
        for i in range(added_count, train_set_count):
            self.train_shapes.append([self.all_shape_matrix[i], self.all_shape_class[i]])
            added_count += 1
        for i in range(added_count, validation_set_count+added_count):
            self.validate_shapes.append([self.all_shape_matrix[i], self.all_shape_class[i]])
            added_count += 1
        for i in range(added_count, test_set_count+added_count):
            self.test_shapes.append([self.all_shape_matrix[i], self.all_shape_class[i]])
            added_count += 1


def main():
    x = 50
    y = 50
    DG = Data_generator(x, 5, y, x, 100)
    print(DG.image_count_in_set)
    DG.generate_split_image_sets(70, 20, 10)
    for i in range(0, 40):
        fig = plt.figure(figsize=(5, 5))
        plt.imshow(DG.all_shape_matrix[i])
        plt.title("Plot 2D array")
    plt.show()


if __name__ == '__main__':
    main()

