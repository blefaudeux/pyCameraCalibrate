#!/usr/bin/env python

"""
Created on Wed Oct 31 14:26:31 2012

@author: Benjamin Lefaudeux (blefaudeux at github)

This script uses OpenCV to calibrate a camera, or a pair of cameras (intrinsic 
and extrinsic parameters). You can use it by running "python camera_calibration.py"
on the command line.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

import os       # File and folders navigation

import cv2
import numpy as np
import sys
import utils
import time

# Import XML
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


class cameraCalibration:
    # "Parameters" field can be used to change some settings
    # namely the automatic pattern validation and the visualisation
    # it should be a tuple of the form :
    # [show-pictures, auto_validation, auto_max_pict_number, auto_save, 'file_path', 
    #   pattern_size(x,y), patch_dimensions(x,y), mono]
    def __init__(self, parameters=None):
        self.obj_points = []
        self.img_points = []

        self.obj_points_l = []
        self.obj_points_r = []

        self.img_points_l = []
        self.img_points_r = []

        self.pictures = []
        self.max_frames_i = 0
        self.n_pattern_found = 0

        self.use_camera = False
        self.intrinsics = []
        self.distorsion = []

        self.frame_size = (0, 0)
        self.frame_size_max = (800, 600)

        self.sq_size_h = 0.0
        self.sq_size_v = 0.0

        self.stereo = None

        if parameters is None:
            self.show_pictures = True
            self.auto_validation = False
            self.auto_save = False
            self.file_path = ''
            self.pattern_size = (0, 0)

        elif isinstance(parameters, tuple):
            self.show_pictures = parameters[0]
            self.auto_validation = parameters[1]
            self.auto_save = parameters[2]

            if parameters[2]:
                self.max_frames_i = -1

            if len(parameters) >= 5 and isinstance(parameters[4], basestring):
                self.file_path = parameters[4]
            else:
                self.file_path = ''

            if len(parameters) >= 6:
                self.pattern_size = parameters[5]

            if len(parameters) >= 7:
                (self.sq_size_h, self.sq_size_v) = parameters[6]

            if len(parameters) >= 8:
                self.stereo = parameters[7]

    def calibrate(self):
        # If calibration type is unkown at this point, ask the user
        if self.stereo is None:
            calib_type = utils.getAnswer('Stereo or Mono calibration ? (s/m) : ', 'sm')

            if calib_type == 's':
                self.stereo = True
            else:
                self.stereo = False

        # Start the appropriate calibration processes
        if self.stereo:
            self.stereoCalibrate()
        else:
            self.monoCalibrate()

    def readFiles(self, folder_path):
        # Read in folders, subfolders,...
        n_files = 0

        # Deal with faulty folder paths
        if folder_path == '':
            folder_path = '.'

        folder_path = os.path.join(folder_path, '')

        # Scan folder and load pictures
        folder_path = os.path.abspath(folder_path)

        if not os.path.isdir(folder_path):
            print "Trouble reading folder path {}".format(folder_path)

        for dirname, dirnames, filenames in os.walk(folder_path):

            filenames = utils.sortNicely(filenames)

            for filename in filenames:
                print "Reading file {}".format(filename)

                full_filepath = os.path.join(folder_path, filename)

                if filename[-3:] == "bmp" or filename[-3:] == "png" or filename[-3:] == "jpg":

                    try:
                        self.pictures.append(cv2.imread(full_filepath, cv2.CV_LOAD_IMAGE_GRAYSCALE))
                        if self.pictures[n_files] is None:
                            self.pictures.pop()
                            print "Error loading file {}".format(filename)
                        else:
                            n_files += 1

                    except ValueError:
                        print "Error loading file {}".format(filename)

        if n_files == 0:
            sys.exit("Could not read any picture")

        if n_files == 0:
            print("Could not read any picture, please correct file path")
            return False

        print "{} pictures read".format(n_files)
        return True

    def chooseCalibrationSettings(self):
        # Get the path where all the files are stored
        if len(self.file_path) == 0:
            file_read = False
            path = None

            while not file_read:
                path = raw_input("Path for the calibration files : ")
                file_read = self.readFiles(path)

            self.file_path = path

        else:
            self.readFiles(self.file_path)

        # Get the pattern dimensions in terms of patch number:
        if self.pattern_size == (0, 0):
            h_dim = utils.getAnswer("Number of inner corners on the horizontal dimension ? ", '12345678910')
            v_dim = utils.getAnswer("Number of inner corners on the vertical dimension ? ", '12345678910')

            # Enter the number of squares over each dimensions
            self.pattern_size = (int(h_dim), int(v_dim))
            print "Chessboard dimensions : {} x {}".format(self.pattern_size[0], self.pattern_size[1])

        # Get every patch dimension :
        if (self.sq_size_h, self.sq_size_v) == (0.0, 0.0):
            get_square_size = False
            while not get_square_size:
                sq_size = raw_input("Horizontal Size (in m) of the squares ? ")

                try:
                    self.sq_size_h = float(sq_size)
                    get_square_size = True

                except ValueError:
                    print "Cannot determine dimension"

            get_square_size = False

            while not get_square_size:
                sq_size = raw_input("Vertical Size (in m) of the squares ? ")

                try:
                    self.sq_size_v = float(sq_size)
                    get_square_size = True

                except ValueError:
                    print "Cannot determine dimension"

        # Get the max number of frames:
        if self.max_frames_i != -1:
            get_max_frames = False
            while not get_max_frames:
                max_frames = raw_input("How many frames ? ")

                try:
                    self.max_frames_i = int(max_frames)
                    get_max_frames = True

                except ValueError:
                    print "Cannot determine max number of frames"

    def recordPattern_cam(self):
        n_frames = 0

        pattern_points = np.zeros((np.prod(self.pattern_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(self.pattern_size).T.reshape(-1, 2)

        cv2.namedWindow("captureStream", cv2.CV_WINDOW_AUTOSIZE)
        cv2.namedWindow("patternDetection", cv2.CV_WINDOW_AUTOSIZE)

        save_files = utils.getAnswer("Would you like to save picture files ? (y/n)", 'yn') == 'y'
        finished_parsing = False

        cam_online = utils.getCam()

        if cam_online:
            while not finished_parsing and (self.max_frames_i == -1 or n_frames < self.max_frames_i):
                success, new_frame = cam_online.read()

                if success:
                    # Convert to B&W (if necessary ?)
                    grey_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
                    found, corners = cv2.findChessboardCorners(grey_frame, self.pattern_size)

                    cv2.imshow("captureStream", new_frame)
                    cv2.waitKey(2)

                    if found:
                        # Refine position
                        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)

                        cv2.cornerSubPix(grey_frame, corners, (11, 11), (-1, -1), term)

                        # Draw detected pattern
                        cv2.drawChessboardCorners(new_frame, self.pattern_size, corners, found)
                        cv2.imshow("patternDetection", new_frame)
                        cv2.waitKey()

                        # Store values
                        self.img_points.append(corners.reshape(-1, 2))
                        self.obj_points.append(pattern_points)

                        n_frames += 1
                        print "{} patterns found".format(n_frames)

                        if save_files:
                            cv2.imwrite("calib_{}.bmp".format(n_frames), grey_frame)
                            cv2.imwrite("calib_{}_pattern.bmp_".format(n_frames), new_frame)

                self.frame_size = (len(new_frame[0]), len(new_frame))

        cv2.destroyAllWindows()

    def recordPattern_files(self):
        # Get patterns on every picture in "pictures[]"
        pattern_points = np.zeros((np.prod(self.pattern_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(self.pattern_size).T.reshape(-1, 2)
        pattern_points[:, 0] = pattern_points[:, 0] * self.sq_size_h
        pattern_points[:, 1] = pattern_points[:, 1] * self.sq_size_v

        n_frames = 0
        n_count = 0
        b_left = False
        b_skip_next = False

        if not self.auto_validation:
            cv2.namedWindow("patternDetection", cv2.CV_WINDOW_AUTOSIZE)
            print "Recording patterns from files, press r to reject"
        else:
            print "Looking for patterns.."

        for new_frame in self.pictures:
            n_count += 1

            new_pict = np.array(new_frame)

            found, corners = cv2.findChessboardCorners(new_pict, self.pattern_size)

            if not found:
                print "Could not find pattern on picture {}".format(n_count)

            if found and not b_skip_next:
                # Refine position
                term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
                cv2.cornerSubPix(new_frame, corners, (11, 11), (-1, -1), term)

                # Draw detected pattern
                cv2.drawChessboardCorners(new_frame, self.pattern_size, corners, found)

                # Resize
                if new_frame.shape[1] > self.frame_size_max[0]:
                    new_size = (int(new_frame.shape[0]/float(new_frame.shape[1]) * self.frame_size_max[0]),
                                self.frame_size_max[0])

                else:
                    new_size = new_frame.shape

                resized_pict = cv2.resize(new_frame, (new_size[1], new_size[0]))

                if not self.auto_validation:
                    # Show and wait for key
                    cv2.imshow("patternDetection", resized_pict)
                    key_choice = cv2.waitKey()
                else:
                    key_choice = -1

                if key_choice == 114:
                    b_reject = True
                    print "Rejected"
                else:
                    b_reject = False

                if not b_reject:
                    # Store values
                    if not self.stereo:
                        self.img_points.append(corners.reshape(-1, 2))
                        self.obj_points.append(pattern_points)

                    else:
                        # Right picture
                        if (n_count % 2) == 0:
                            self.img_points_r.append(corners.reshape(-1, 2))
                            self.obj_points_r.append(pattern_points)
                            b_skip_next = False     # Should be useless
                            b_left = False

                        # Left picture
                        else:
                            self.img_points_l.append(corners.reshape(-1, 2))
                            self.obj_points_l.append(pattern_points)
                            b_skip_next = False    # Don't skip next picture
                            b_left = True

                    n_frames += 1

                else:
                    # Picture is rejected. If left picture, skip next one also
                    if (n_count % 2) == 1:
                        print "Left picture missed, next picture is skipped"
                        b_skip_next = True

                    # If right picture, remove alose the previous left one
                    if b_left:
                        print "Right picture missed, removing previous left picture"
                        self.img_points_l.pop()
                        self.obj_points_l.pop()
                        n_frames -= 1

                if n_frames == 1:
                    print "One pattern found"
                else:
                    print "{} patterns found".format(n_frames)

                self.frame_size = (len(new_frame[0]), len(new_frame))

            elif self.stereo:
                if (n_count % 2) == 1:
                    # Left pattern is missed, skip next pattern
                    print "Left pattern missed, skipping next frame"
                    b_skip_next = True

                elif b_skip_next:
                    print "Right pattern skipped because left was missed"
                    b_skip_next = False

                else:
                    print "Right pattern is missed, removing previous frame"
                    self.img_points_l.pop()
                    self.obj_points_l.pop()
                    n_frames -= 1
                    b_skip_next = False

            if (self.max_frames_i != -1) and (n_frames >= self.max_frames_i):
                print "Enough grabbed frames"
                break

        if n_frames == 0:
            print "Could not find any pattern"
            return 0

        return n_frames

    def recordPatterns(self):
        if self.use_camera:
            self.recordPattern_cam()
            self.n_pattern_found = len(self.img_points)

        else:
            self.n_pattern_found = self.recordPattern_files()

        if self.n_pattern_found == 0:
            sys.exit("Calibration could not finish")

        print "Patterns collection finished, starting calibration"

    def stereoCalibrate(self):
        print "If you calibrate from files, make sure stereo files are \n in the left-right-left-right order"

        # Get settings & files
        self.stereo = True
        self.chooseCalibrationSettings()

        # Get patterns
        self.recordPatterns()

        # Compute the intrisic parameters first :
        rvecs = [np.zeros(3, dtype=np.float32) for _ in xrange(self.max_frames_i)]
        tvecs = [np.zeros(3, dtype=np.float32) for _ in xrange(self.max_frames_i)]

        self.intrinsics.append(np.zeros((4, 4), dtype=np.float32))
        self.intrinsics.append(np.zeros((4, 4), dtype=np.float32))

        self.distorsion.append(np.zeros(8, dtype=np.float32))
        self.distorsion.append(np.zeros(8, dtype=np.float32))

        # Call OpenCV routines to do the dirty work
        print "Computing intrisic parameters for the first camera"
        res = cv2.calibrateCamera(self.obj_points_l, self.img_points_l, self.frame_size,
                                  self.intrinsics[0], self.distorsion[0], rvecs, tvecs)

        rms, self.intrinsics[0], self.distorsion[0], rvecs, tvecs = res

        print "Computing intrisic parameters for the second camera"
        res = cv2.calibrateCamera(self.obj_points_r, self.img_points_r, self.frame_size,
                                  self.intrinsics[1], self.distorsion[1], rvecs, tvecs)

        rms, self.intrinsics[1], self.distorsion[1], rvecs, tvecs = res

        # Compute calibration parameters :
        print "Calibrating cameras.."
        print "Frame size : {}".format(self.frame_size)

        # set stereo flags
        stereo_flags = 0
        #        stereo_flags |= cv2.CALIB_FIX_INTRINSIC
        stereo_flags |= cv2.CALIB_USE_INTRINSIC_GUESS     # Refine intrinsic parameters
        #        stereo_flags |= cv2.CALIB_FIX_PRINCIPAL_POINT     # Fix the principal points during the optimization.
        #        stereo_flags |= cv2.CALIB_FIX_FOCAL_LENGTH        # Fix focal length
        #        stereo_flags |= cv2.CALIB_FIX_ASPECT_RATIO        # fix aspect ratio
        stereo_flags |= cv2.CALIB_SAME_FOCAL_LENGTH       # Use same focal length
        #        stereo_flags |= cv2.CALIB_ZERO_TANGENT_DIST       # Set tangential distortion to zero
        #        stereo_flags |= cv2.CALIB_RATIONAL_MODEL
        #  Use 8 param rational distortion model instead of 5 param plumb bob model

        res = cv2.stereoCalibrate(self.obj_points_l,
                                  self.img_points_l, self.img_points_r,
                                  self.frame_size,
                                  self.intrinsics[0], self.distorsion[0],
                                  self.intrinsics[1], self.distorsion[1])

        (rms, int_left, dist_left, int_right, dist_right, R, T, E, F) = res

        # Output
        print "Calibration done. Residual RMS : {}".format(rms)

        if rms > 1.0:
            print "Calibration looks faulty, please re-run"

        print "\nCalibration parameters : \n Intrinsics -left \n {} \n\n Distorsion -left\n {}".format(
            int_left, dist_left)

        print "\nCalibration parameters : \n Intrinsics -right \n {} \n\n Distorsion -right\n {}".format(
            int_right, dist_right)

        print "\nRotation : \n{}\n".format(R)

        print "\nTranslation : \n{}\n".format(T)

        print "\nEssential matrix : \n{}\n".format(E)  # Essential matrix

        print "\nFundamental matrix : \n{}\n".format(F)  # Fundamental matrix

        # TODO : Compute perspective matrix !

        # Save calibration parameters
        save_file = utils.getAnswer("Would you like to save the results ? (y/n) ", 'yn')

        b_write_success = False

        if save_file == "y":
            while not b_write_success:
                save_XML = utils.getAnswer("Save in XML format ? (y/n) ", "yn")

                filepath = raw_input("Where do you want to save the file ? (enter file path) ")

                try:
                    if save_XML:
                        calib_file = utils.handlePath(filepath, "camera_calibration.xml")
                        self.saveParametersXML(R, T, calib_file)

                    else:
                        file_left = utils.handlePath(filepath, "_left.txt")
                        self.saveParameters(int_left, dist_left, R, T, file_left, self.n_pattern_found)

                        file_right = utils.handlePath(filepath, "_right.txt")
                        self.saveParameters(int_right, dist_right, R, T, file_right, self.n_pattern_found)

                        print "Parameters file written"
                        b_write_success = True
                except:
                    print "Wrong path, please correct"

                time.sleep(2)

        return

    def monoCalibrate(self):
        # Get settings
        self.chooseCalibrationSettings()

        # Get patterns
        self.recordPatterns()

        # Compute intrinsic parameters
        rvecs = [np.zeros(3) for _ in xrange(self.max_frames_i)]    # Rotation and translation matrix
        tvecs = [np.zeros(3) for _ in xrange(self.max_frames_i)]

        _obj_points = np.array(self.obj_points, dtype=np.float32)
        _img_points = np.array(self.img_points, dtype=np.float32)
        self.intrinsics.append(np.zeros((4, 4), dtype=np.float32))
        self.distorsion.append(np.zeros(8, dtype=np.float32))

        rms, self.intrinsics[0], self.distorsion[0], _rvecs, _tvecs = cv2.calibrateCamera(_obj_points, _img_points,
                                                                                          self.frame_size,
                                                                                          self.intrinsics[0],
                                                                                          self.distorsion[0],
                                                                                          rvecs, tvecs)

        print "Calibration done"
        np.set_printoptions(precision=2)
        np.set_printoptions(suppress=True)

        print "Residual RMS (pxl units) :\n"
        print(rms)

        print "\nRotations :\n"
        print(rvecs)

        print "\nTranslations :\n"
        print(tvecs)

        print "\nCalibration parameters : Intrinsics \n"
        print(self.intrinsics)

        print "Distorsion \n"
        print(self.distorsion)

        print "\nNumber of pictures used : {}".format(self.n_pattern_found)

        # Save calibration parameters
        if not self.auto_save:
            save_file = utils.getAnswer("Would you like to save the results ? (y/n) ", 'yn')
            b_write_success = False

            if save_file == "y":
                while not b_write_success:
                    filepath = raw_input("Where do you want to save the file ? (enter file path) ")
                    filepath = utils.handlePath(filepath, "calib_results")

                    try:
                        self.saveParameters(rvecs, tvecs, rms, filepath + '.txt')
                        self.saveParametersJSON(rvecs, tvecs, filepath + '.json')
                        b_write_success = True

                    except ValueError:
                        print "Wrong path, please correct"

                    time.sleep(2)

        else:
            calib_file_path = utils.handlePath(self.file_path, "calib_results")
            self.saveParameters(rvecs, tvecs, rms, calib_file_path + '.txt')
            self.saveParametersJSON(rvecs, tvecs, calib_file_path + '.json')
            print "Saved calibration file"

        return

    def saveParameters(self, rotation, translation, rms, path):
        with open(path, "w") as FILE:
            # Write parameters :
            FILE.write("Calibration error (pixels) over {} pictures: \n".format(self.n_pattern_found))
            FILE.write("{}\n\n".format(rms))

            FILE.write("Intrisic Matrix : \n")
            FILE.write("{}\n\n".format(self.intrinsics))

            FILE.write("Distorsion coefficients :\n")
            FILE.write("{}\n\n".format(self.distorsion))

            FILE.write("Rotations :\n")
            FILE.write("{}\n\n".format(rotation))

            FILE.write("Translations :\n")
            FILE.write("{}\n\n".format(translation))

            FILE.write("Pattern used : \n")
            FILE.write("{} squares\n".format(self.pattern_size))
            FILE.write("{}m x {}m \n".format(self.sq_size_h, self.sq_size_v))

    def saveParametersXML(self, rotation, translation, path):
        # Build XML structure from the settings to be saved
        cam_calibration = ET.Element('camera_calibration')
        cam_mat_0 = ET.SubElement(cam_calibration, 'camera_matrix_0')
        dist_mat_0 = ET.SubElement(cam_calibration, 'dist_matrix_0')

        if self.stereo:
            dist_mat_1 = ET.SubElement(cam_calibration, 'dist_matrix_1')
            cam_mat_1 = ET.SubElement(cam_calibration, 'camera_matrix_1')

        rot_mat = ET.SubElement(cam_calibration, 'rotation_matrix')
        tr_mat = ET.SubElement(cam_calibration, 'translation_matrix')
        pict_size = ET.SubElement(cam_calibration, 'picture_size')

        # Fill camera matrices (intrinsics)
        cam_mat_0.set('a0', str(self.intrinsics[0][0, 0]))
        cam_mat_0.set('a1', str(self.intrinsics[0][0, 1]))
        cam_mat_0.set('a2', str(self.intrinsics[0][0, 2]))
        cam_mat_0.set('b0', str(self.intrinsics[0][1, 0]))
        cam_mat_0.set('b1', str(self.intrinsics[0][1, 1]))
        cam_mat_0.set('b2', str(self.intrinsics[0][1, 2]))
        cam_mat_0.set('c0', str(self.intrinsics[0][2, 0]))
        cam_mat_0.set('c1', str(self.intrinsics[0][2, 1]))
        cam_mat_0.set('c2', str(self.intrinsics[0][2, 2]))

        if self.stereo:
            cam_mat_1.set('a0',str(self.intrinsics[1][0, 0]))
            cam_mat_1.set('a1',str(self.intrinsics[1][0, 1]))
            cam_mat_1.set('a2',str(self.intrinsics[1][0, 2]))
            cam_mat_1.set('b0',str(self.intrinsics[1][1, 0]))
            cam_mat_1.set('b1',str(self.intrinsics[1][1, 1]))
            cam_mat_1.set('b2',str(self.intrinsics[1][1, 2]))
            cam_mat_1.set('c0',str(self.intrinsics[1][2, 0]))
            cam_mat_1.set('c1',str(self.intrinsics[1][2, 1]))
            cam_mat_1.set('c2',str(self.intrinsics[1][2, 2]))

        # Fill in distorsion matrix
        dist_mat_0.set('k1', str(self.distorsion[0][0]))
        dist_mat_0.set('k2', str(self.distorsion[0][1]))
        dist_mat_0.set('p1', str(self.distorsion[0][2]))
        dist_mat_0.set('p2', str(self.distorsion[0][3]))
        dist_mat_0.set('k3', str(self.distorsion[0][4]))

        if self.stereo:
            dist_mat_1.set('k1', str(self.distorsion_r[0]))
            dist_mat_1.set('k2', str(self.distorsion_r[1]))
            dist_mat_1.set('p1', str(self.distorsion_r[2]))
            dist_mat_1.set('p2', str(self.distorsion_r[3]))
            dist_mat_1.set('k3', str(self.distorsion_r[4]))

        # Fill in rotation matrix
        rot_mat.set('a0', str(rotation[0, 0]))
        rot_mat.set('a1', str(rotation[0, 1]))
        rot_mat.set('a2', str(rotation[0, 2]))
        rot_mat.set('b0', str(rotation[1, 0]))
        rot_mat.set('b1', str(rotation[1, 1]))
        rot_mat.set('b2', str(rotation[1, 2]))
        rot_mat.set('c0', str(rotation[2, 0]))
        rot_mat.set('c1', str(rotation[2, 1]))
        rot_mat.set('c2', str(rotation[2, 2]))

        # Fill in translation matrix
        tr_mat.set('a0', str(translation[0, 0]))
        tr_mat.set('a1', str(translation[1, 0]))
        tr_mat.set('a2', str(translation[2, 0]))

        # Fill in picture size
        pict_size.set('width', str(self.frame_size[0]))
        pict_size.set('height', str(self.frame_size[1]))

        # Write to file
        with ET.ElementTree(cam_calibration) as tree:
            tree.write(path)

    def saveParametersJSON(self, rotation, translation, path):
        # Fill in the dict object first
        calib_results = {'intrinsics': [], 'distorsion': [], 'rotation': [], 'translation': [], 'picture_size': []}

        # Intrinsics
        if len(self.intrinsics) > 0:
            if self.stereo:
                calib_results['intrinsics'] = [[], []]

                for _, item in enumerate(self.intrinsics[0]):
                    for _, i in enumerate(item):
                        calib_results['intrinsics'][0].append(i)

                for _, item in enumerate(self.intrinsics[1]):
                    for _, i in enumerate(item):
                        calib_results['intrinsics'][1].append(i)

            else:
                for _, item in enumerate(self.intrinsics[0]):
                    for _, i in enumerate(item):
                        calib_results['intrinsics'].append(i)

        # Distorsion
        if len(self.distorsion) > 0:
            if self.stereo:
                calib_results['distorsion'] = [[], []]

                for _, item in enumerate(self.distorsion[0]):
                    calib_results['distorsion'][0].append(item[0])

                for _, item in enumerate(self.distorsion[1]):
                    calib_results['distorsion'][1].append(item[0])

            else:
                for _, item in enumerate(self.distorsion[0]):
                    calib_results['distorsion'].append(str(item))

        calib_results['picture_size'] = [self.frame_size[0], self.frame_size[1]]

        # Motion matrices and we're done
        for _, item in enumerate(rotation):
            for _, i in enumerate(item):
                calib_results['rotation'].append(i)

        for _, item in enumerate(translation):
            for _, i in enumerate(item):
                calib_results['translation'].append(i)

        # Use the standard JSON dump
        import json
        with open(path, 'w') as fp:
            json.dump(calib_results, fp)

    def loadJSON(self, path):
        import json
        with open(path, 'r') as fp:
            calib_params = json.load(fp)

        # TODO: Parse the dictionary

