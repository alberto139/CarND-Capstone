from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import cv2
from collections import Counter
import object_class
import os

class TLClassifier(object):
    def __init__(self):
        model_path = os.getcwd() + "/light_classification/frozen_inference_graph.pb" 
        self.conf_threshold = 0.5

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

        

        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()

            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        with detection_graph.as_default():
            #self.sess = tf.Session()
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            self.tensor_dict = {}
            for key in ['num_detections','detection_boxes','detection_scores','detection_classes']:
                tensor_name = key + ":0"
                if tensor_name in all_tensor_names:
                    self.tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

            self.image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
        print("DONE with TLC INIT")

    def get_classification(self, img):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #print("getting classification")
        output_dict = self.sess.run(
        self.tensor_dict, feed_dict={
        self.image_tensor: np.expand_dims(img, 0)})
        output_dict['num_detections'] = int(
        output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]

        # Detection
        detections = []
        for i, element in enumerate(output_dict['detection_boxes']):
            if output_dict['detection_scores'][i] > self.conf_threshold:
                temp_object = object_class.Object(img, output_dict['detection_boxes'][i], output_dict['detection_classes'][i])
                detections.append(temp_object)


        # Get subimages for traffic lights
        detected_colors = []
        for i, tl in enumerate(detections):
            tl.subimg = tl.frame[tl.ymin:tl.ymax, tl.xmin:tl.xmax]
            #cv2.imshow("tl" + str(i), tl.subimg)
            hsv = cv2.cvtColor(tl.subimg, cv2.COLOR_BGR2HSV)
            mask_green = cv2.inRange(hsv, (36, 100, 100), (70, 255,255)) # Green
            mask_yellow = cv2.inRange(hsv, (15, 180, 40), (35, 255,255)) # Yellow
            mask_red1 = cv2.inRange(hsv, (0, 70, 50), (10, 255,255)) # Red
            mask_red2 = cv2.inRange(hsv, (170, 70, 50), (180, 255,255)) # Red2
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)

            green = (sum(sum(mask_green)))
            yellow = (sum(sum(mask_yellow)))
            red = (sum(sum(mask_red)))
            colors = [red, yellow, green]
            max_color = colors.index(max(colors))
            detected_colors.append(max_color)
    

        state = -1
        if detected_colors:
            #state = mode(detected_colors)
            count = Counter(detected_colors)
            
            state = count.most_common(1)[0][0]
            print("TL STATE: " + str(state))
            #return state

        return state
