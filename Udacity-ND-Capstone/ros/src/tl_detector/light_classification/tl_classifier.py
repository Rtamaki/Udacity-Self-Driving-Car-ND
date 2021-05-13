#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import tf
import cv2
import yaml
from scipy.spatial import KDTree
import numpy as np
import cv2
import os
import keras

POSITIVE_COLORS_THRESHOLD = 100


class TLClassifier(object):
    def __init__(self):
        #TODO load classifier

        # Load Keras model
        self.directory_path = os.path.dirname(os.path.abspath(__file__))
        self.model = None
        pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction

        traffic_color = TrafficLight.UNKNOWN

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


        # define red color range
        lower_red_1 = np.array([0,100,100])
        upper_red_1 = np.array([10,255,255])

        lower_red_2 = np.array([170,100,100])
        upper_red_2 = np.array([179,255,255])

        mask1 = cv2.inRange(hsv_image, lower_red_1, upper_red_1)
        mask2 = cv2.inRange(hsv_image, lower_red_2, upper_red_2)
        # rospy.logwarn('\nTest\n')
        # Say that the right is red if at least of the pixels are within the given range
        # rospy.logwarn('Sum {0}\n'.format(np.sum(mask)))

        if (np.sum(mask1) + np.sum(mask2) > POSITIVE_COLORS_THRESHOLD):
            rospy.logwarn('--- RED ---')
            traffic_color = TrafficLight.RED
        else:
            rospy.logwarn('--- GREEN ---')
            traffic_color = TrafficLight.GREEN


        return traffic_color





    # def get_classification(self, image):
    #     """Determines the color of the traffic light in the image
    #
    #     Args:
    #         image (cv::Mat): image containing the traffic light
    #
    #     Returns:
    #         int: ID of traffic light color (specified in styx_msgs/TrafficLight)
    #
    #     """
    #     #TODO implement light color prediction
    #
    #     traffic_color = TrafficLight.UNKNOWN
    #
    #     # pass
    #
    #     if not self.model:
    #         self.model = keras.models.load_model(self.directory_path + '/model7.h5')
    #
    #
    #     # rospy.logwarn('\nTest\n')
    #     # Say that the right is red if at least of the pixels are within the given range
    #
    #     # Get the traffic light color and its likehood
    #     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #     image_array = np.asarray(hsv)
    #     prediction = self.model.predict(image_array[None, :, :, :], batch_size=1)
    #     print('-----------------')
    #     print(prediction)
    #     traffic_color = np.argmax(prediction)
    #     probability = np.amax(prediction)
    #
    #     # Say it is green unless we are pretty sure it is red
    #     # rospy.logwarn('Prediction ' + str(traffic_color) + ' - ' + str(probability))
    #     if (traffic_color == 0):
    #         traffic_color = TrafficLight.RED
    #     else:
    #         traffic_color = TrafficLight.GREEN
    #
    #
    #     return traffic_color


    # Function to create data to train CNN
    def save_image(self, image, light_state):


        time = rospy.get_time()
        path = './' + str(round(light_state)) + str(time) + '.png'
        cv2.imwrite(path, image)
