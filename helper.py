import cv2 
import csv
import numpy as np
from random import shuffle

class CSVPreprocessor(object):
    """
    Class that processes the csv file from the simulator into a dataset ready format
    """
    def __init__(self, input_csv, correction_factor=0, validation_split=0):
        """
        :param input_csv: The input csv file from the simulator
        :param correction_factor: Correction factor to apply to the left and right images
        :param validation_split: Percentage of data to use as the validation set
        """
        self.input_csv = input_csv 
        self.correction_factor = correction_factor
        self.validation_split = validation_split
    
    
    def _get_filename(self, path):
        """
        Get the filename from a path string
        :param path: full or partial path of the file using "/" for directory division
        :type path: str 
        :returns filename
        """
        return path.split("/")[-1]


    def _split_row(self, row):
        """
        Split a single row of our data into 3 rows with an img_path: angle key/value pair
        :param row: Row to split into 3 rows
        :returns: list of dicts with 3 img: angle key/value pair
        """

        left_img = self._get_filename(row[0])
        center_img = self._get_filename(row[1])
        right_img = self._get_filename(row[2])
        
        # Compute new angle values by applying the correction factor to each image
        center_angle = float(row[3])
        left_angle = center_angle + self.correction_factor
        right_angle = center_angle - self.correction_factor
        
        return [
            [ left_img, left_angle ],
            [ center_img, center_angle ],
            [ right_img, right_angle ]
        ]
   
    
    def preprocess(self, training_output="train.csv", 
                         validation_output="validation.csv",
                         shuffle_data=True):
        """
        Parse our raw csv file into so that each row is an image with its associated steering angle
        :param training_output: The output csv file for the training dataset
        :param validation_output: The output csv file for the validation dataset
        :param validation_split: The percentage split of all examples to put in the validation set 
        :param shuffle_data: Wether or not to shuffle the data before splitting and writing
        :returns: Tuple of the training and validation csv filename written to disk
        """
        print("Preprocessing CSV file...")
        with open(self.input_csv, "r") as f:
            reader = csv.reader(f)
            rows = [row for row in reader]
            n_examples = len(rows)
            n_validation = int(n_examples * self.validation_split)
            n_train = n_examples - n_validation

            # Shuffle data if requested
            if shuffle_data:
                shuffle(rows)

            # Training examples
            with open(training_output, "w") as w:
                writer = csv.writer(w)
                for i in range(n_train):
                    writer.writerows(self._split_row(rows[i]))

            # Validation examples
            with open(validation_output, "w") as w:
                writer = csv.writer(w)
                for i in range(n_train, n_examples):
                    writer.writerows(self._split_row(rows[i]))

        return (training_output, validation_output) 

class CSVImageDataGen(object):
    def __init__(self, training_csv, validation_csv, img_dir="", batch_size=64):
        self.img_dir= img_dir
        self.training_data = self._csv_to_array(training_csv)
        self.validation_data = self._csv_to_array(validation_csv)
        self.img_shape = self.get_img_shape()
        self.img_height, self.img_width, self.img_depth = self.img_shape


    def _get_img(self, filename):
        """
        Load the image from a specified path
        :param filename: Path of the image to load
        :param top_path: Path of the directory of the data (Optional)
        :returns: Image as a numpy array
        """
        return cv2.imread(self.img_dir + filename)
    

    def _csv_to_array(self, csv_file):
        """
        Loads a preprocessed csv file to RAM as an iterable of image filename and associated angle
        :param csv_file: Path of preprocessed csv file 
        :returns: Array of arrays (filename, angle)
        """
        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            return [row for row in reader]

    
    def read_row(self, row):
        X = np.expand_dims(self._get_img(row[0]), axis=0)
        y = np.array([row[1]])
        return (X, y)

    def get_img_shape(self):
        return self._get_img(self.training_data[0][0]).shape


    def training_generator(self):
        for example in self.training_data:
            yield self.read_row(example) 
   

    def validation_generator(self):
        for example in self.validation_data:
            yield self.read_row(example) 
