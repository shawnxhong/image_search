"""
 Copyright (c) 2018-2024 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import logging as log
import os
import os.path as osp

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine

from .face_detector import FaceDetector


class FacesDatabase:
    IMAGE_EXTENSIONS = ['jpg', 'png']

    class Identity:
        def __init__(self, label, descriptors):
            self.label = label
            self.descriptors = descriptors

        @staticmethod
        def cosine_dist(x, y):
            return cosine(x, y) * 0.5

    def __init__(self, path, face_identifier, landmarks_detector, face_detector=None, no_show=False):
        path = osp.abspath(path)
        self.fg_path = path
        self.no_show = no_show
        paths = []
        if osp.isdir(path):
            paths = [osp.join(path, f) for f in os.listdir(path)
                      if f.split('.')[-1] in self.IMAGE_EXTENSIONS]
        else:
            log.error("Wrong face images database path. Expected a "
                      "path to the directory containing %s files, "
                      "but got '%s'" %
                      (" or ".join(self.IMAGE_EXTENSIONS), path))

        if len(paths) == 0:
            log.error("The images database folder has no images.")

        self.database = []
        for path in paths:
            label = osp.splitext(osp.basename(path))[0]
            image = cv2.imread(path, flags=cv2.IMREAD_COLOR)

            if face_detector:
                rois = face_detector.infer((image,))
                if len(rois) < 1:
                    log.warning("Not found faces on the image '{}'".format(path))
                    continue
                if len(rois) > 1:
                    log.warning("Found {} faces on '{}'. Using the largest one for label '{}'".format(len(rois), path, label))
                    rois = [max(rois, key=lambda roi: roi.size[0] * roi.size[1])]
            else:
                w, h = image.shape[1], image.shape[0]
                rois = [FaceDetector.Result([0, 0, 0, 0, 0, w, h])]

            for roi in rois:
                r = [roi]
                landmarks = landmarks_detector.infer((image, r))

                face_identifier.start_async(image, r, landmarks)
                descriptor = face_identifier.get_descriptors()[0]

                log.debug("Adding label {} to the gallery".format(label))
                self.add_item(descriptor, label)

    def match_faces(self, descriptors, match_algo='HUNGARIAN'):
        database = self.database
        distances = np.empty((len(descriptors), len(database)))
        for i, desc in enumerate(descriptors):
            for j, identity in enumerate(database):
                dist = []
                for id_desc in identity.descriptors:
                    dist.append(FacesDatabase.Identity.cosine_dist(desc, id_desc))
                distances[i][j] = dist[np.argmin(dist)]

        matches = []
        if match_algo == 'MIN_DIST':
            for i in range(len(descriptors)):
                id = np.argmin(distances[i])
                min_dist = distances[i][id]
                matches.append((id, min_dist))
        else:
            _, assignments = linear_sum_assignment(distances)
            for i in range(len(descriptors)):
                if len(assignments) <= i:
                    matches.append((0, 1.0))
                    continue

                id = assignments[i]
                distance = distances[i, id]
                matches.append((id, distance))

        return matches

    def create_new_label(self, path, id):
        while osp.exists(osp.join(path, "face{}.jpg".format(id))):
            id += 1
        return "face{}".format(id)

    def check_if_face_exist(self, desc, threshold):
        match = -1
        for j, identity in enumerate(self.database):
            dist = []
            for id_desc in identity.descriptors:
                dist.append(FacesDatabase.Identity.cosine_dist(desc, id_desc))
            if dist[np.argmin(dist)] < threshold:
                match = j
                break
        return match

    def check_if_label_exists(self, label):
        match = -1
        import re
        name = re.split(r'-\d+$', label)
        if not len(name):
            return -1, label
        label = name[0].lower()

        for j, identity in enumerate(self.database):
            if identity.label == label:
                match = j
                break
        return match, label

    def dump_faces(self, image, desc, name):
        match, label = self.add_item(desc, name)
        if match < 0:
            filename = "{}-0.jpg".format(label)
            match = len(self.database)-1
        else:
            filename = "{}-{}.jpg".format(label, len(self.database[match].descriptors)-1)
        filename = osp.join(self.fg_path, filename)

        log.debug("Dumping image with label {} and path {} on disk.".format(label, filename))
        if osp.exists(filename):
            log.warning("File with the same name already exists at {}. So it won't be stored.".format(self.fg_path))
        cv2.imwrite(filename, image)
        return match

    def add_item(self, desc, label):
        match = -1
        if not label:
            label = self.create_new_label(self.fg_path, len(self.database))
            log.warning("Trying to store an item without a label. Assigned label {}.".format(label))
        else:
            match, label = self.check_if_label_exists(label)

        if match < 0:
            self.database.append(FacesDatabase.Identity(label, [desc]))
        else:
            self.database[match].descriptors.append(desc)
            log.debug("Appending new descriptor for label {}.".format(label))

        return match, label

    def __getitem__(self, idx):
        return self.database[idx]

    def __len__(self):
        return len(self.database)
