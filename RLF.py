import argparse
import os
import numpy as np
from scipy.sparse import lil_matrix as matrix
from PIL import Image, ImageOps 

class RLF:
    """Run Length Features"""
    GRAY_LEVEL_CHANNEL_MAX_VALUE = 255

    # horizontal, vertical, diagonal 1, diagonal 2
    run_directions = [True,True,True,True]
    num_of_gray_levels = 0
    lut = []

    def __init__(self, num_of_gray_levels = 8):
        self.num_of_gray_levels = num_of_gray_levels
        self.create_lut()
        self.parameters = dict.fromkeys(['SRE', 'LRE', 'GLU', 'RLU', 'LGRE', 'HGRE', 'SRLGE', 'SRHGE', 'LRLGE', 'LRHGE', 'RP'])

    def set_run_directions(self, directions):
        self.run_directions = directions

    def set_num_of_gray_levels(self, num_of_gray_levels):
        self.num_of_gray_levels = num_of_gray_levels
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

    def load_image(self, img_name):
        """ Processes input image so every pixel has value <0, num_ranges) based on its gray level """
        with Image.open(img_name) as img:
            img_grayscale = ImageOps.grayscale(img)
            self.img_width, self.img_height = img.size
            
            self.processed_img = img_grayscale.point(lambda p: self.lut[p])
            self.GLRLM = None
            self.K = None

    def generate_GLRLM_horizontal(self, GLRLM_size):
        #GLRLM(i,j) - Gray-Level Run Length Matrix for direction, where i (row) is features gray-level and j (column) is its length
        GLRLM = np.zeros(GLRLM_size, dtype=np.int64)

        last_pixel_value = self.processed_img.getpixel((0,0))
        feature_length = 1
            
        for y in range(0, self.img_height):
            for x in range(1, self.img_width):
                pixel_value = self.processed_img.getpixel((x, y))
                if pixel_value == last_pixel_value:
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

    def generate_GLRLM_vertical(self, GLRLM_size):
        #GLRLM(i,j) - Gray-Level Run Length Matrix for direction, where i (row) is features gray-level and j (column) is its length
        GLRLM = np.zeros(GLRLM_size, dtype=np.int64)

        last_pixel_value = self.processed_img.getpixel((0,0))
        feature_length = 1
            
        for x in range(0, self.img_width):
            for y in range(1, self.img_height):
                pixel_value = self.processed_img.getpixel((x, y))
                if pixel_value == last_pixel_value:
                    feature_length += 1
                else :
                    GLRLM[last_pixel_value][feature_length] += 1
                    feature_length = 1
                    last_pixel_value = pixel_value

            GLRLM[last_pixel_value][feature_length] += 1
            feature_length = 1

            if x + 1 < self.img_width:
                last_pixel_value = self.processed_img.getpixel((x+1, 0))

        return GLRLM

    def generate_GLRLM_diagonal_left_to_right(self, GLRLM_size):
        #GLRLM(i,j) - Gray-Level Run Length Matrix for direction, where i (row) is features gray-level and j (column) is its length
        GLRLM = np.zeros(GLRLM_size, dtype=np.int64)
            
        for x_outer in range(0, self.img_width):
            x = x_outer            
            y = 0
            
            feature_length = 1
            last_pixel_value = self.processed_img.getpixel((x, y))
            x += 1
            y += 1

            while  x < self.img_width and y < self.img_height:
                pixel_value = self.processed_img.getpixel((x, y))
                if pixel_value == last_pixel_value:
                    feature_length += 1
                else :
                    GLRLM[last_pixel_value][feature_length] += 1
                    feature_length = 1
                    last_pixel_value = pixel_value
                x += 1
                y += 1

            GLRLM[last_pixel_value][feature_length] += 1

        # lower left bitmap triangle (not covered by main loop)
        for y_outer in range(1, self.img_height):
            y = y_outer
            x = 0

            feature_length = 1
            last_pixel_value = self.processed_img.getpixel((x,y))
            x += 1
            y += 1

            while x < self.img_width and y < self.img_height:
                pixel_value = self.processed_img.getpixel((x, y))
                if pixel_value == last_pixel_value:
                    feature_length += 1
                else :
                    GLRLM[last_pixel_value][feature_length] += 1
                    feature_length = 1
                    last_pixel_value = pixel_value
                x += 1
                y += 1
                
            GLRLM[last_pixel_value][feature_length] += 1

        return GLRLM
    
    def generate_GLRLM_diagonal_right_to_left(self, GLRLM_size):
        #GLRLM(i,j) - Gray-Level Run Length Matrix for direction, where i (row) is features gray-level and j (column) is its length
        GLRLM = np.zeros(GLRLM_size, dtype=np.int64)
            
        for x_outer in range(self.img_width - 1, -1, -1):
            x = x_outer            
            y = 0
            
            feature_length = 1
            last_pixel_value = self.processed_img.getpixel((x, y))
            x -= 1
            y += 1

            while  x > 0 and y < self.img_height:
                pixel_value = self.processed_img.getpixel((x, y))
                if pixel_value == last_pixel_value:
                    feature_length += 1
                else :
                    GLRLM[last_pixel_value][feature_length] += 1
                    feature_length = 1
                    last_pixel_value = pixel_value
                x -= 1
                y += 1

            GLRLM[last_pixel_value][feature_length] += 1

        # lower left bitmap triangle (not covered by main loop)
        for y_outer in range(1, self.img_height):
            y = y_outer
            x = self.img_width - 1

            feature_length = 1
            last_pixel_value = self.processed_img.getpixel((x,y))
            x -= 1
            y += 1

            while x < self.img_width and y < self.img_height:
                pixel_value = self.processed_img.getpixel((x, y))
                if pixel_value == last_pixel_value:
                    feature_length += 1
                else :
                    GLRLM[last_pixel_value][feature_length] += 1
                    feature_length = 1
                    last_pixel_value = pixel_value
                x -= 1
                y += 1
                
            GLRLM[last_pixel_value][feature_length] += 1

        return GLRLM

    def generate_GLRLM(self):
        """Calculates GLRLM matrix from features in directions specified in parameter"""
        #GLRLM(i,j) - Gray-Level Run Length Matrix for direction, where i (row) is features gray-level and j (column) is its length
        GLRLM_size = (self.num_of_gray_levels, max([self.img_width+1, self.img_height+1]))
        self.GLRLM = np.zeros(GLRLM_size, dtype=np.int64)

        if (self.run_directions[0]):
            self.GLRLM += self.generate_GLRLM_horizontal(GLRLM_size)
        if (self.run_directions[1]):
            self.GLRLM += self.generate_GLRLM_vertical(GLRLM_size)
        if (self.run_directions[2]):
            self.GLRLM += self.generate_GLRLM_diagonal_left_to_right(GLRLM_size)
        if (self.run_directions[3]):
            self.GLRLM += self.generate_GLRLM_diagonal_right_to_left(GLRLM_size)

        self.K = self.GLRLM.sum()

    def write_RLF_parameters_to_file(self):
        None

    def generate_paramters(self):
        self.generate_GLRLM()

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

        # GLU - Gray Level Uniformity
        sum = 0
        for i in range(self.num_of_gray_levels):
            inside_sum = 0
            for j in range(1, self.img_width):
                inside_sum += self.GLRLM[i][j]
            sum += inside_sum * inside_sum
        parameters['GLU'] = sum / self.K

        # RLU - Run Length Uniformity
        sum = 0
        for j in range(1, self.img_width):
            inside_sum = 0
            for i in range(self.num_of_gray_levels):
                inside_sum += self.GLRLM[i][j]
            sum += inside_sum * inside_sum
        parameters['RLU'] = sum / self.K

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

    def csv_insert_header(self, file_name):
        # erase existing file contents
        open(file_name, 'w').close()
        with open(file_name, 'a') as file:
            file.write('image_name')
            file.write(''.join([f', {k}' for k, _ in self.parameters.items()]))
            file.write('\n')


    def save_params_to_csv(self, file_name, image_name):
        with open(file_name, 'a') as file:
            # write header row       
            file.write(f'')
            params_csv = ''.join(['{:.2f}, '.format(p) for _, p in self.parameters.items()])
            file.write(f'{image_name}, {params_csv}\n')


if __name__ == '__main__':
    run_directions_choices = ['horizontal', 'vertical', 'diagonal_left_to_right', 'diagonal_right_to_left', "all"]

    program_description="""
    Program calculating Run Length Feature method parameters for imges.
    """

    parser = argparse.ArgumentParser(description=program_description)
    parser.add_argument('-n', '--num-of-ranges', type=int, default='8', help='number of ranges into which pixel is classified based on its gray level value. default is 8.')
    parser.add_argument('-d', '--direction-of-runs', choices=run_directions_choices, default='all', nargs='*', help='directions in which runs will be performed')
    parser.add_argument('-c', '--csv_output', type=str, help='output parameters to csv file')
    parser.add_argument('-p', '--print_output', action='store_true', help='Prints parameter values for each image. Default is true.')
    parser.add_argument('IMG', type=str, help='Path to image, or directory containing images.')
    args = parser.parse_args()
    
    rlf = RLF(args.num_of_ranges)
    # if run_directions is not set to all, set rlf's directions
    if 'all' not in args.direction_of_runs:
        rlf.set_run_directions([d in args.direction_of_runs for d in run_directions_choices[:4]])

    if os.path.isdir(args.IMG):
        # directory of image files
        if args.csv_output:
            rlf.csv_insert_header(args.csv_output)

        for image in os.listdir(args.IMG):
            rlf.load_image(f'{args.IMG}/{image}')
            rlf.generate_paramters()

            if args.csv_output:
                rlf.save_params_to_csv(args.csv_output, image)
            if args.print_output:
                print(f'\n{image}')
                print(rlf.parameters)

    elif os.path.isfile(args.IMG):
        # single image file
        rlf.load_image(args.IMG)
        rlf.generate_paramters()

        if args.csv_output:
            rlf.csv_insert_header(args.csv_output)
            rlf.save_params_to_csv(args.csv_output, args.IMG)
        if args.print_output:
            print(f'\n{args.IMG}')
            print(rlf.parameters)
            
    else:
        print('Given IMG is not a image file or a directory')  