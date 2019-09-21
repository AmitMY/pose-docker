import os
from functools import lru_cache

import tensorflow as tf
import numpy as np
import cv2
from tqdm import tqdm
import argparse
import json

from draw import draw_pose

from sys import path

path.append("cpm/")

from utils import cpm_utils
from models.nets import cpm_hand_slim


class HandDetector(object):
    def __init__(self, model_path, img_size, stages=6, joints=21, use_kalman=True,
                 kalman_noise=3e-2, color_channel='RGB', cmap_radius=21, hmap_size=46,
                 tf_device='/cpu:0'):

        print("Device", tf_device)
        self.model_path = model_path
        self.input_size = img_size
        self.stages = stages
        self.joints = joints
        self.use_kalman = use_kalman
        self.kalman_noise = kalman_noise
        self.color_channel = color_channel
        self.cmap_radius = cmap_radius
        self.hmap_size = hmap_size
        self.tf_device = tf_device
        with tf.device(self.tf_device):
            """Build graph
            """
            if self.color_channel == 'RGB':
                input_data = tf.placeholder(dtype=tf.float32, shape=[None, self.input_size, self.input_size, 3],
                                            name='input_image')
            else:
                input_data = tf.placeholder(dtype=tf.float32, shape=[None, self.input_size, self.input_size, 1],
                                            name='input_image')

            center_map = tf.placeholder(dtype=tf.float32, shape=[None, self.input_size, self.input_size, 1],
                                        name='center_map')

            self.model = cpm_hand_slim.CPM_Model(self.stages, self.joints + 1)
            self.model.build_model(input_data, center_map, 1)

        saver = tf.train.Saver()

        """Create session and restore weights
        """
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        self.sess.run(tf.global_variables_initializer())
        if self.model_path.endswith('pkl'):
            self.model.load_weights_from_file(self.model_path, self.sess, False)
        else:
            saver.restore(self.sess, self.model_path)

        self.test_center_map = cpm_utils.gaussian_img(self.input_size, self.input_size, self.input_size / 2,
                                                      self.input_size / 2,
                                                      self.cmap_radius)
        self.test_center_map = np.reshape(self.test_center_map, [1, self.input_size, self.input_size, 1])

        # Check weights
        for variable in tf.trainable_variables():
            with tf.variable_scope('', reuse=True):
                var = tf.get_variable(variable.name.split(':0')[0])

        # Create kalman filters
        if self.use_kalman:
            self.kalman_filter_array = [cv2.KalmanFilter(4, 2) for _ in range(self.joints)]
            for _, joint_kalman_filter in enumerate(self.kalman_filter_array):
                joint_kalman_filter.transitionMatrix = np.array(
                    [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
                    np.float32)
                joint_kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
                joint_kalman_filter.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                                               np.float32) * self.kalman_noise
        else:
            self.kalman_filter_array = None

    def get_locations(self, test_img, stage_heatmap_np):
        last_heatmap = stage_heatmap_np[len(stage_heatmap_np) - 1][0, :, :, 0:self.joints].reshape(
            (self.hmap_size, self.hmap_size, self.joints))
        last_heatmap = cv2.resize(last_heatmap, (test_img.shape[1], test_img.shape[0]))
        locations = []

        # Plot joint colors
        if self.kalman_filter_array is not None:
            for joint_num in range(self.joints):
                joint_coord = np.unravel_index(np.argmax(last_heatmap[:, :, joint_num]),
                                               (test_img.shape[0], test_img.shape[1]))
                joint_coord = np.array(joint_coord).reshape((2, 1)).astype(np.float32)
                self.kalman_filter_array[joint_num].correct(joint_coord)
                locations.append((joint_coord[1][0], joint_coord[0][0]))
        else:
            for joint_num in range(self.joints):
                joint_coord = np.unravel_index(np.argmax(last_heatmap[:, :, joint_num]),
                                               (test_img.shape[0], test_img.shape[1]))
                locations.append((joint_coord[1][0], joint_coord[0][0]))
        return locations

    def predict(self, test_img):
        """
        Using the pre-trained model to predict the figers locations.
        The model predicts the location of 21 anchor points on the
        given hand, the basis of the hand, and additional 4 points
        along each finger.
        This function returns only the anchor and the tip of fingers
        in this order: (anchor, thumb, index finger, middle finger,
        ring finger, pinky). For these names see:
            https://en.wiktionary.org/wiki/ring_finger
        Each location is a tuple, where the first index represents
        the x value and the second represents the y value.
        NOTICE: x values starts counting from buttom left.
                y values starts counting from TOP left.
        """

        assert test_img.shape == (self.input_size, self.input_size, 3)

        with tf.device(self.tf_device):
            if self.color_channel == 'GRAY':
                test_img = np.dot(test_img[..., :3], [0.299, 0.587, 0.114]).reshape(
                    (self.input_size, self.input_size, 1))

            test_img_input = test_img / 256.0 - 0.5
            test_img_input = np.expand_dims(test_img_input, axis=0)

            # print(test_img_input)

            # Inference
            predict_heatmap, stage_heatmap_np = self.sess.run([self.model.current_heatmap,
                                                               self.model.stage_heatmap,
                                                               ],
                                                              feed_dict={'input_image:0': test_img_input,
                                                                         'center_map:0': self.test_center_map})

            # Get finger anchors from detection.
            locations = self.get_locations(test_img, stage_heatmap_np)

            # filtering the relevant anchor locations.
            relevant_locations = [locations[0], locations[4], locations[8], locations[12], locations[16], locations[20]]
            # print(locations)
            return relevant_locations, locations

    def predict_batch(self, input_test_img):
        with tf.device(self.tf_device):
            test_img = cpm_utils.read_image(input_test_img, [], self.input_size, 'IMAGE')

            test_img_resize = cv2.resize(test_img, (self.input_size, self.input_size))

            if self.color_channel == 'GRAY':
                test_img_resize = np.dot(test_img_resize[..., :3], [0.299, 0.587, 0.114]).reshape(
                    (self.input_size, self.input_size, 1))

            test_img_input = test_img_resize / 256.0 - 0.5
            test_img_input = np.expand_dims(test_img_input, axis=0)

            test_img_input = np.repeat(test_img_input, 100, 0)

            # Inference
            predict_heatmap, stage_heatmap_np = self.sess.run([self.model.current_heatmap,
                                                               self.model.stage_heatmap,
                                                               ],
                                                              feed_dict={'input_image:0': test_img_input,
                                                                         'center_map:0': self.test_center_map})

            # Get finger anchors from detection.
            locations = self.get_locations(test_img, stage_heatmap_np)

            # filtering the relevant anchor locations.
            relevant_locations = [locations[0], locations[4], locations[8], locations[12], locations[16], locations[20]]
            # print(locations)
            return relevant_locations

    def predict_batch_updated(self, input_batch, right_hand=True):
        """
        Function that uses a pre-trained model for predicting fingers location
        from images.

        Input:
            input_batch: a numpy array, with 4 dimensions: (batch size, x, y, rgb)
            right_hand: a boolean telling if the hands are right or left.
                        In the case they are left hands, flipping the images.

        Returns:
            a batched version of the following values:
            relevant locations: indices of 6 major location points of the hand on the image.
            locations: all of the detected locations of the hand.
        """
        with tf.device(self.tf_device):
            if not right_hand:
                input_batch = np.flip(input_batch, axis=2)

            input_batch = input_batch / 256.0 - 0.5

            # Inference
            predict_heatmap, stage_heatmap_np = self.sess.run([self.model.current_heatmap,
                                                               self.model.stage_heatmap,
                                                               ],
                                                              feed_dict={'input_image:0': input_batch,
                                                                         'center_map:0': self.test_center_map})

            # Get finger anchors from detection.
            result = np.array(stage_heatmap_np)
            locations_batch = [self.get_locations(input_batch[i], np.expand_dims(result[:, i], 1))
                               for i in range(result.shape[1])]

            # Case integer
            locations_batch = [[(int(x), int(y)) for (x, y) in locations] for locations in locations_batch]

            if not right_hand:
                locations_batch = [[(self.input_size - x, y) for (x, y) in locations] for locations in locations_batch]

            # locations = self.get_locations(input_batch, stage_heatmap_np)

            # filtering the relevant anchor locations.
            relevant_locations = [
                [locations[0], locations[4], locations[8], locations[12], locations[16], locations[20]]
                for locations in locations_batch]
            # print(locations)
            return list(zip(relevant_locations, locations_batch))


# We only want to load one instance.
@lru_cache()
def get_hand_detector(device='gpu'):
    model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../cpm/models/model.pkl')
    return HandDetector(model_path, 368, tf_device=device + ':0')


def batch(iterable, n=1):
    length = len(iterable)
    for ndx in range(0, length, n):
        yield iterable[ndx:min(ndx + n, length)]


def hand_pose(images, right_hand=True, mode="RGB", batch_size=100, device='gpu'):
    hd = get_hand_detector(device)
    if mode == "RGB":
        images = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in images]

    locations = []
    for b in tqdm(list(batch(images, batch_size)), unit="batches-" + str(batch_size)):
        batch_frames = np.stack(b, axis=0)
        locations += hd.predict_batch_updated(batch_frames, right_hand=right_hand)

    return locations


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help="Point to a directory containing images and nothing else", required=True)
    parser.add_argument('-o', '--output', help='Point to a directory where you want the output to be saved',
                        required=True)
    parser.add_argument('-f', '--format', default='json', help='One of image/json')
    parser.add_argument('-d', '--device', default='gpu', help='One of cpu or gpu')
    args = parser.parse_args()

    files = os.listdir(args.input)

    if len(files) == 0:
        raise Exception("Input directory is empty")

    print("Found", len(files), "files")

    images = [cpm_utils.read_image(os.path.join(args.input, f), [], 368, 'IMAGE') for f in files]

    poses = [pose for _, pose in hand_pose(images, right_hand=True, mode="BGR", device=args.device)]

    for f, image, pose in zip(files, images, poses):
        save_loc = os.path.join(args.output, f)
        if args.format == "json":
            json.dump(pose, open(save_loc + ".json", "w"))
        else:
            new_image = draw_pose(image, pose)
            cv2.imwrite(save_loc, new_image)

    print(poses)
