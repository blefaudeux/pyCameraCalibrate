#!/usr/bin/python2

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

import os  # File and folders navigation
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np

import utils


def find_pattern(raw_pict, pattern_size):
    gray = cv2.cvtColor(raw_pict, cv2.COLOR_BGR2GRAY)
    pict = np.array(gray)
    found, corners = cv2.findChessboardCorners(pict, pattern_size)

    if found:
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        cv2.cornerSubPix(pict, corners, (11, 11), (-1, -1), term)
        cv2.drawChessboardCorners(raw_pict, pattern_size, corners, found)

    return [found, corners, pict, raw_pict]  # Let's hope that we take refs of those, no copies..


def show_pattern(window, max_size, pict):
    new_size = pict.shape
    if pict.shape[1] > max_size:
        new_size = (int(pict.shape[0] / float(pict.shape[1]) * max_size),
                    max_size[0])

    resized_pict = cv2.resize(pict, (new_size[1], new_size[0]))
    cv2.imshow(window, resized_pict)  # Show and wait for key
    key = cv2.waitKey()
    return key


class CameraCalibrationSettings:
    def __init__(self):
        self.file_path = ''

        self.show_pictures = False
        self.auto_validation = False
        self.auto_save = True
        self.stereo = False
        self.use_camera = False

        self.pattern_size = (0, 0)
        self.sq_size_h = 0.0
        self.sq_size_v = 0.0

        self.max_frames_i = -1


class CameraCalibration:
    def __init__(self, parameters=CameraCalibrationSettings()):
        self.pict_names = []
        self.obj_points = []
        self.img_points = []

        self.obj_points_l = []
        self.obj_points_r = []

        self.img_points_l = []
        self.img_points_r = []

        self.pictures = []
        self.n_pattern_found = 0

        self.intrinsics = []
        self.distorsion = []

        self.frame_size = (0, 0)
        self.frame_size_max = (800, 600)

        if isinstance(parameters, CameraCalibrationSettings):
            self.params = parameters

    def calibrate(self):
        if self.params.stereo:
            self._calibrate_stereo()
        else:
            self._calibrate_mono()

    def _read_files(self, folder_path):
        n_files = 0

        # Scan folder and load pictures
        folder_path = os.path.abspath(folder_path)

        if not os.path.isdir(folder_path):
            print("Trouble reading folder path {}".format(folder_path))

        self.pict_names = []
        for dirname, _, filenames in os.walk(folder_path):

            filenames = utils.sortNicely(filenames)

            for filename in filenames:
                print "Reading file {}".format(filename)

                full_filepath = os.path.join(dirname, filename)

                if filename[-3:] == "bmp" or filename[-3:] == "png" or filename[-3:] == "jpg":

                    try:
                        self.pictures.append(cv2.imread(full_filepath))
                        if self.pictures[n_files] is None:
                            self.pictures.pop()
                            print " {}".format(filename)
                        else:
                            self.pict_names.append(full_filepath)
                            n_files += 1

                    except ValueError:
                        print " {}".format(filename)

            break   # Just read one level

        if n_files == 0:
            sys.exit("Could not read any picture")

        if n_files == 0:
            print "Could not read any picture, please correct file path"
            return False

        print "{} pictures read".format(n_files)
        return True

    def _choose_calibration_settings(self):
        # Get the path where all the files are stored
        if len(self.params.file_path) == 0:
            file_read = False
            path = None

            while not file_read:
                path = raw_input("Path for the calibration files : ")
                file_read = self._read_files(path)

            self.params.file_path = path

        else:
            self._read_files(self.params.file_path)

        # Get the pattern dimensions in terms of patch number:
        if self.params.pattern_size == (0, 0):
            h_dim = utils.getAnswer("Number of inner corners on the horizontal dimension ? ", '12345678910')
            v_dim = utils.getAnswer("Number of inner corners on the vertical dimension ? ", '12345678910')

            # Enter the number of squares over each dimensions
            self.params.pattern_size = (int(h_dim), int(v_dim))
            print "Chessboard dimensions : {} x {}".format(self.params.pattern_size[0], self.params.pattern_size[1])

        # Get every patch dimension :
        if (self.params.sq_size_h, self.params.sq_size_v) == (0.0, 0.0):
            get_square_size = False

            while not get_square_size:
                sq_size = raw_input("Horizontal Size (in m) of the squares ? ")

                try:
                    self.params.sq_size_h = float(sq_size)
                    get_square_size = True

                except ValueError:
                    print "Cannot determine dimension"

            get_square_size = False

            while not get_square_size:
                sq_size = raw_input("Vertical Size (in m) of the squares ? ")

                try:
                    self.params.sq_size_v = float(sq_size)
                    get_square_size = True

                except ValueError:
                    print "Cannot determine dimension"

        # Get the max number of frames:
        if self.params.max_frames_i != -1:
            get_max_frames = False
            while not get_max_frames:
                max_frames = raw_input("How many frames ? ")

                try:
                    self.params.max_frames_i = int(max_frames)
                    get_max_frames = True

                except ValueError:
                    print "Cannot determine max number of frames"

    def _record_pattern_cam(self):
        n_frames = 0

        pattern_points = np.zeros((np.prod(self.params.pattern_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(self.params.pattern_size).T.reshape(-1, 2)

        cv2.namedWindow("captureStream", cv2.CV_WINDOW_AUTOSIZE)
        cv2.namedWindow("patternDetection", cv2.CV_WINDOW_AUTOSIZE)

        save_files = utils.getAnswer("Would you like to save picture files ? (y/n)", 'yn') == 'y'
        finished_parsing = False

        cam_online = utils.getCam()

        if cam_online:
            while not finished_parsing and (self.params.max_frames_i == -1 or n_frames < self.params.max_frames_i):
                success, new_frame = cam_online.read()

                if success:
                    # Convert to B&W (if necessary ?)
                    grey_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
                    found, corners = cv2.findChessboardCorners(grey_frame, self.params.pattern_size)

                    cv2.imshow("captureStream", new_frame)
                    cv2.waitKey(2)

                    if found:
                        # Refine position
                        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)

                        cv2.cornerSubPix(grey_frame, corners, (11, 11), (-1, -1), term)

                        # Draw detected pattern
                        cv2.drawChessboardCorners(new_frame, self.params.pattern_size, corners, found)
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

    def _record_pattern_files(self):
        # Get patterns on every picture
        pattern_points = np.zeros((np.prod(self.params.pattern_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(self.params.pattern_size).T.reshape(-1, 2)
        pattern_points[:, 0] = pattern_points[:, 0] * self.params.sq_size_h
        pattern_points[:, 1] = pattern_points[:, 1] * self.params.sq_size_v

        n_frames = 0
        n_count = 0
        b_left = False
        b_skip_next = False

        pool = Pool()
        results = []

        print "Starting pattern research"
        for pic in self.pictures:
            results.append(pool.apply_async(find_pattern, args=(pic, self.params.pattern_size)))

        pool.close()
        pool.join()

        # Validate
        if not self.params.auto_validation:
            cv2.namedWindow("patternDetection", cv2.CV_WINDOW_AUTOSIZE)
            print "Recording patterns from files, press r to reject, other to accept"
        else:
            print "Automatic pattern validation"

        for res in results:
            found, corners, pict, raw_pict = res.get()

            n_count += 1

            if not found:
                print "Could not find pattern on picture {}".format(n_count)

            if found and not b_skip_next:
                if not self.params.auto_validation:
                    key_choice = show_pattern("patternDetection", self.frame_size_max, raw_pict)
                else:
                    key_choice = -1

                if key_choice != 114:
                    if not self.params.stereo:
                        self.img_points.append(corners.reshape(-1, 2))
                        self.obj_points.append(pattern_points)

                    else:
                        b_skip_next = False  # Should be useless

                        # Right picture
                        if (n_count % 2) == 0:
                            self.img_points_r.append(corners.reshape(-1, 2))
                            self.obj_points_r.append(pattern_points)
                            b_left = False

                        # Left picture
                        else:
                            self.img_points_l.append(corners.reshape(-1, 2))
                            self.obj_points_l.append(pattern_points)
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
                    print "1 pattern found"
                else:
                    print "{} patterns found".format(n_frames)

                self.frame_size = (len(raw_pict[0]), len(raw_pict))

            elif self.params.stereo:
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

            if (self.params.max_frames_i != -1) and (n_frames >= self.params.max_frames_i):
                print "Enough grabbed frames"
                break

        if n_frames == 0:
            print "Could not find any pattern"
            return 0

        return n_frames

    def _record_patterns(self):
        if self.params.use_camera:
            self._record_pattern_cam()
            self.n_pattern_found = len(self.img_points)

        else:
            self.n_pattern_found = self._record_pattern_files()

        if self.n_pattern_found == 0:
            sys.exit("Calibration could not finish")

        print "Patterns collection finished, starting calibration"

    def _calibrate_stereo(self):
        print "-- If you calibrate from files, make sure stereo files are \n in the left-right-left-right order --"

        # Get settings & files
        self._choose_calibration_settings()

        # Get patterns
        self._record_patterns()

        # Compute the intrisic parameters first :
        rvecs = [np.zeros(3, dtype=np.float32) for _ in xrange(self.params.max_frames_i)]
        tvecs = [np.zeros(3, dtype=np.float32) for _ in xrange(self.params.max_frames_i)]

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
        print "Calibrating cameras.. \nFrame size : {}".format(self.frame_size)

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

        (rms, int_left, dist_left, int_right, dist_right, rotations, translations, essential, fundamental) = res

        # Output
        print "Calibration done. Residual RMS : {}".format(rms)

        if rms > 1.0:
            print "Calibration looks faulty, please re-run"

        print "\nCalibration parameters : \n Intrinsics -left \n {} \n\n Distorsion -left\n {}".format(
            int_left, dist_left)

        print "\nCalibration parameters : \n Intrinsics -right \n {} \n\n Distorsion -right\n {}".format(
            int_right, dist_right)

        print "\nRotation : \n{}\n".format(rotations)
        print "\nTranslation : \n{}\n".format(translations)
        print "\nEssential matrix : \n{}\n".format(essential)
        print "\nFundamental matrix : \n{}\n".format(fundamental)

        # Save calibration parameters
        save_file = utils.getAnswer("Would you like to save the results ? (y/n) ", 'yn') == 'y'

        b_write_success = False

        if save_file:
            while not b_write_success:
                filepath = raw_input("Where do you want to save the file ? (enter file path) ")

                try:
                    file_left = utils.handlePath(filepath, "_overall.txt")
                    self._json_save(rotations, translations,  rms, file_left)

                    print "Parameters file written"
                    b_write_success = True

                except IOError:
                    print "Wrong path, please correct"

                time.sleep(2)

        return

    def _calibrate_mono(self):
        self._choose_calibration_settings()
        self._record_patterns()

        # Compute intrinsic parameters
        rvecs = [np.zeros(3) for _ in xrange(self.params.max_frames_i)]    # Rotation and translation matrix
        tvecs = [np.zeros(3) for _ in xrange(self.params.max_frames_i)]

        _obj_points = np.array(self.obj_points, dtype=np.float32)
        _img_points = np.array(self.img_points, dtype=np.float32)
        self.intrinsics.append(np.zeros((4, 4), dtype=np.float32))
        self.distorsion.append(np.zeros(8, dtype=np.float32))

        ret = cv2.calibrateCamera(_obj_points, _img_points, self.frame_size, self.intrinsics[0], self.distorsion[0],
                                  rvecs, tvecs)

        [rms, self.intrinsics[0], self.distorsion[0], _, _] = ret

        print "Calibration done"
        np.set_printoptions(precision=2)
        np.set_printoptions(suppress=True)

        print "Residual RMS (pxl units) :\n{}".format(rms)
        print "\nRotations :\n {} \nTranslations :\n {}".format(rvecs, tvecs)
        print "\nCalibration parameters : Intrinsics \n {}".format(self.intrinsics[0])
        print "Distorsion \n {}".format(self.distorsion[0])
        print "\nNumber of pictures used : {}".format(self.n_pattern_found)

        # Save calibration parameters
        if not self.params.auto_save:
            b_write_success = False

            if utils.getAnswer("Would you like to save the results ? (y/n) ", 'yn') == "y":
                while not b_write_success:
                    filepath = raw_input("Where do you want to save the file ? (enter file path) ")
                    filepath = utils.handlePath(filepath, "calib_results")

                    try:
                        self._json_save(rvecs, tvecs, rms, filepath + '.json')
                        b_write_success = True

                    except ValueError:
                        print "Wrong path, please correct"

        else:
            calib_file_path = utils.handlePath(self.params.file_path, "calib_results")
            self._json_save(rvecs, tvecs, rms, calib_file_path + '.json')
            print "Saved calibration file"

        # Test the distorsion parameters:
        if utils.getAnswer("Would you like to test the distorsion parameters ? (y/n)", 'yn') == 'y':
            root_folder = os.path.dirname(self.pict_names[0])
            dist_folder = os.path.join(root_folder, 'undistorted')
            if not os.path.exists(dist_folder):
                os.makedirs(dist_folder)

            for pict, name in zip(self.pictures, self.pict_names):
                undistorted = self._undistort_frame(pict)
                pict_name = os.path.basename(name)
                cv2.imwrite(os.path.join(dist_folder, pict_name[:-3] + 'undistorted.jpg'), undistorted)
                print "Image {} undistorted".format(pict_name)

        return

    def _undistort_frame(self, frame):
        if self.params.stereo:
            print "[Undistort] Cannot handle stereo for now"
            return None

        return cv2.undistort(frame, self.intrinsics[0], self.distorsion[0])

    def _json_save(self, rotation, translation, rms, path):
        # Fill in the dict object first
        calib_results = {
            'intrinsics': [],
            'distorsion': [],
            'rotation': [],
            'translation': [],
            'picture_size': [],
            'parameters':
            {
                'number_of_pictures': self.n_pattern_found,
                'residue': rms,
                'pattern_shape': self.params.pattern_size,
                'pattern_size': [self.params.sq_size_h, self.params.sq_size_v]
            }
        }

        # Intrinsics
        if len(self.intrinsics) > 0:
            if self.params.stereo:
                for _, item in enumerate(self.intrinsics[0]):
                    calib_results['intrinsics']['left'].append(list(item))

                for _, item in enumerate(self.intrinsics[1]):
                    calib_results['intrinsics']['right'].append(list(item))

            else:
                for _, item in enumerate(self.intrinsics[0]):
                    calib_results['intrinsics'].append(list(item))

        # Distorsion
        if len(self.distorsion) > 0:
            if self.params.stereo:
                for _, item in enumerate(self.distorsion[0]):
                    calib_results['distorsion']['left'].append(item[0])

                for _, item in enumerate(self.distorsion[1]):
                    calib_results['distorsion']['right'].append(item[0])

            else:
                for _, item in enumerate(self.distorsion[0]):
                    calib_results['distorsion'].append(str(item))

        calib_results['picture_size'] = [self.frame_size[0], self.frame_size[1]]

        # Motion matrices and we're done
        for _, rot in enumerate(rotation):
            calib_results['rotation'].append(list(rot))

        for _, trans in enumerate(translation):
            calib_results['translation'].append(list(trans))

        # Use the standard JSON dump
        import json
        with open(path, 'w') as fp:
            json.dump(calib_results, fp, sort_keys=True, indent=4, separators=(',', ': '))


if __name__ == '__main__':
    new_instance = CameraCalibration()
    new_instance.calibrate()
