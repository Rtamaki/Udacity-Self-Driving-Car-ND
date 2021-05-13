import numpy as np
import cv2
import time
import imageio
imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip
from lesson_functions import *



# def video_pipeline(image):
#     classifier = SVM.SVM_classifier()
#     window_sizes = [1.0, 1.25, 1.5, 2.0]
#     search_regions = [[360, 500], [360, 600], [360, 720], [360, 720]]
#     search_config = {'window_sizes': window_sizes, 'search_regions': search_regions}
#     threshold = 1
#     out_img, bboxes = search_image(image, classifier, search_config)
#     out_img = heatmap_pipeline(image, bboxes, threshold)
#     return out_img



def process_video(input_video_name, output_video_name):

    video_pipeline = video_pipe()
    input_video = VideoFileClip(input_video_name)#.subclip(20,40)
    output_video = input_video.fl_image(video_pipeline.process_video)
    output_video.write_videofile(output_video_name, audio=False)








input_video_name = 'project_video.mp4'
output_video_name = 'output.mp4'
process_video(input_video_name, output_video_name)




