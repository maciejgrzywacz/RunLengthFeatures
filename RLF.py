# import click
import argparse
import math
import numpy as np
from scipy.sparse import lil_matrix as matrix
from PIL import Image, ImageOps 

class RLF:
    """Run Length Features"""

    DEBUG_INFO = True
    GRAY_LEVEL_CHANNEL_MAX_VALUE = 255

    # horizontal, vertical, diagonal 1, diagonal 2
    run_directions = [True,True,False,False]
    num_of_gray_levels = 0
    max_feature_length = 5
    range_step = 0


    lut = []
    boundry_levels = []

    def __init__(self, num_of_gray_levels):
        self.num_of_gray_levels = num_of_gray_levels
        self.range_step = self.GRAY_LEVEL_CHANNEL_MAX_VALUE / self.num_of_gray_levels
        self.create_lut()

    def create_lut(self):
        """ Create lut (lookup table) based on number of levels used in classification 

            method used in creating lut makes ranges of length n or n-1, where n = gray_levels / num_of_gray_levels
            There is always: num_of_gray_levels_of_length_n > num_of_gray_levels_of_length_n-1 
        """
        step = self.GRAY_LEVEL_CHANNEL_MAX_VALUE / self.num_of_gray_levels
        next_boundry = step
        level_value = 0
        for i in range(self.GRAY_LEVEL_CHANNEL_MAX_VALUE + 1):
            if (i >= step):
                level_value += 1
                step = next_boundry + step
    
            self.lut.append(level_value)

        if self.lut[255] == self.num_of_gray_levels:
            self.lut[255] = self.num_of_gray_levels - 1

        assert len(self.lut) == 256

    def load_image(self, img_name):
        """ Processes input image so every pixel has value <0, num_ranges) based on its gray level """
        with Image.open(img_name) as img:
            img_grayscale = ImageOps.grayscale(img)
            self.img_width, self.img_height = img.size
            
            self.processed_img = img_grayscale.point(lambda p: self.lut[p])

    def generate_GLRLM_horizontal(self):
        #GLRLM(i,j) - Gray-Level Run Length Matrix for direction, where i (row) is features gray-level and j (column) is its length
        GLRLM = np.zeros((self.num_of_gray_levels, self.img_width), dtype=np.int64)

        last_pixel_value = self.processed_img.getpixel((0,0))
        feature_length = 1
            
        for y in range(0, self.img_height):
            for x in range(1, self.img_width):
                pixel_value = self.processed_img.getpixel((x, y))
                if feature_length < self.max_feature_length - 1 and pixel_value == last_pixel_value:
                    feature_length += 1
                else :
                    GLRLM[last_pixel_value][feature_length] += 1
                    feature_length = 1
                    last_pixel_value = pixel_value

            GLRLM[last_pixel_value][feature_length] += 1
            feature_length = 1

            if y + 1 < self.img_height:
                last_pixel_value = self.processed_img.getpixel((0, y+1))

        return GLRLM

    def generate_GLRLM_vertical(self):
        #GLRLM(i,j) - Gray-Level Run Length Matrix for direction, where i (row) is features gray-level and j (column) is its length
        GLRLM = np.zeros((self.num_of_gray_levels, self.img_width), dtype=np.int64)

        last_pixel_value = self.processed_img.getpixel((0,0))
        feature_length = 1
            
        for x in range(0, self.img_width):# img_width
            for y in range(1, self.img_height):
                pixel_value = self.processed_img.getpixel((x, y))
                if feature_length < self.max_feature_length - 1 and pixel_value == last_pixel_value:
                    feature_length += 1
                else :
                    GLRLM[last_pixel_value][feature_length] += 1
                    feature_length = 1
                    last_pixel_value = pixel_value

            GLRLM[last_pixel_value][feature_length] += 1
            feature_length = 1

            if x + 1 < self.img_width: #img_width
                last_pixel_value = self.processed_img.getpixel((x+1, 0))

        return GLRLM
        None

    def generate_GLRLM_diagonal_left_to_right(self):
        None
    
    def generate_GLRLM_diagonal_right_to_left(self):
        None

    def generate_GLRLM(self):
        """Calculates GLRLM matrix from features in directions specified in parameter"""
        self.GLRLM = np.zeros((self.num_of_gray_levels, self.img_width), dtype=np.int64)

        if (self.run_directions[0]):
            self.GLRLM += self.generate_GLRLM_horizontal()
        if (self.run_directions[1]):
            self.GLRLM += self.generate_GLRLM_vertical()
        if (self.run_directions[2]):
            # TODO: add diagonal run
            None
        if (self.run_directions[3]):
            # TODO: add diagonal run
            None

        self.K = self.GLRLM.sum()

    def write_RLF_parameters_to_file(self):
        None

    def generate_paramters(self):
        #GLRLM(i,j) - Gray-Level Run Length Matrix for direction, where i (row) is features gray-level and j (column) is its length
        parameters = dict()

        # SRE - short run emphasis
        sum = 0
        for i in range(self.num_of_gray_levels):
            for j in range(1, self.img_width):
                sum += self.GLRLM[i][j] / (j*j)
        parameters['SRE'] = sum / self.K

        # LRE - long run emhasis
        sum = 0
        for i in range(self.num_of_gray_levels):
            for j in range(1, self.img_width):
                sum += self.GLRLM[i][j] * j*j
        parameters['LRE'] = sum / self.K

        # LGRE
        sum = 0
        for i in range(self.num_of_gray_levels):
            for j in range(1, self.img_width):
                sum += self.GLRLM[i][j] / ((i+1)*(i+1))
        parameters['LGRE'] = sum / self.K

        # HGRE
        sum = 0
        for i in range(self.num_of_gray_levels):
            for j in range(1, self.img_width):
                sum += self.GLRLM[i][j] * (i+1)*(i+1)
        parameters['HGRE'] = sum / self.K

        # SRLGE
        sum = 0
        for i in range(self.num_of_gray_levels):
            for j in range(1, self.img_width):
                sum += self.GLRLM[i][j] / ((i+1)*(i+1) * j*j)
        parameters['SRLGE'] = sum / self.K

        # SRHGE
        sum = 0
        for i in range(self.num_of_gray_levels):
            for j in range(1, self.img_width):
                sum += self.GLRLM[i][j] * ((i+1)*(i+1)) / (j*j)
        parameters['SRHGE'] = sum / self.K

        # LRLGE
        sum = 0
        for i in range(self.num_of_gray_levels):
            for j in range(1, self.img_width):
                sum += self.GLRLM[i][j] * j*j / ((i+1)*(i+1))
        parameters['LRLGE'] = sum / self.K

        # LRHGE
        sum = 0
        for i in range(self.num_of_gray_levels):
            for j in range(1, self.img_width):
                sum += self.GLRLM[i][j] * ((i+1)*(i+1)) * j*j
        parameters['LRHGE'] = sum / self.K

        # GLU - Gray Level Uniformity
        sum = 0
        for i in range(self.num_of_gray_levels):
            inside_sum = 0
            for j in range(1, self.img_width):
                inside_sum += self.GLRLM[i][j]
            sum += inside_sum * inside_sum
        parameters['GLNUf'] = sum / self.K

        # RLU - Run Length Uniformity
        sum = 0
        for j in range(1, self.img_width):
            inside_sum = 0
            for i in range(self.num_of_gray_levels):
                inside_sum += self.GLRLM[i][j]
            sum += inside_sum * inside_sum
        parameters['RLNU'] = sum / self.K

        # RP - Run Percentage
        sum = 0
        for i in range(self.num_of_gray_levels):
            for j in range(1, self.img_width):
                sum += self.GLRLM[i][j]
        parameters['RP'] = self.K / sum
        
        self.parameters = parameters
        return parameters

    def save_params(self, file_name):
        with open(file_name, 'w') as file:
            for param_name in self.parameters:
                file.write(f'{param_name}: {self.parameters[param_name]:.2f}\n')


if __name__ == '__main__':

    # args = argparse.ArgumentParser(description='Run Length Features for images.')
    # args.add_argument('--num-of-ranges', type=int, help='number of gray levels')
    # args.add_argument('--direction-of-runs', choices=['horizontal', 'vertical', 'diagonal_left_to_right', 'diagonal_right_to_left', "all"], default='all', nargs='*', help='directions in which runs will be performed')
    
    # arguments = args.parse_args()

    # TODO: parse arguments 
    # 1: -n/-num-of-ranges <int>
    # 2: -d/--direction-of-features [horizontal, vertical, diagonal_left_up_to_right_down, diafonal_right_up_to_left_down, all(default)]
    
    num_of_gray_levels = 7

    rlf = RLF(num_of_gray_levels)
    rlf.load_image('blurry_image.jpg')
    rlf.generate_GLRLM()
    image_params = rlf.generate_paramters()
    print(image_params)
    rlf.save_params('blurry.txt')

    
    rlf.load_image('buildings_image.jpg')
    rlf.generate_GLRLM()
    image_params = rlf.generate_paramters()
    print(image_params)
    rlf.save_params('buildings.txt')