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
        self.intrinsics = np.zeros((3, 3), dtype=np.float32)
        self.intrinsics_l = np.zeros((3, 3), dtype=np.float32)
        self.intrinsics_r = np.zeros((3, 3), dtype=np.float32)

        self.distorsion = np.zeros(8, dtype=np.float32)
        self.distorsion_l = np.zeros(8, dtype=np.float32)
        self.distorsion_r = np.zeros(8, dtype=np.float32)

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
        if len(self.file_path) ==0:
            file_read = False
    
            while not file_read:
                path = raw_input("Path for the calibration files : ")
                file_read = self.readFiles(path)
                
            self.file_path = path
            
        else:
            file_read = self.readFiles(self.file_path)

        # Get the pattern dimensions in terms of patch number:
        if self.pattern_size == (0, 0):
            h_dim = utils.getAnswer( "Number of inner corners on the horizontal dimension ? ", '12345678910')
    
            v_dim = utils.getAnswer( "Number of inner corners on the vertical dimension ? ", '12345678910')
    
            # Enter the number of squares over each dimensions
            self.pattern_size = (int(h_dim), int(v_dim))
            print "Chessboard dimensions : {} x {}"\
                .format(self.pattern_size[0], self.pattern_size[1])
            
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

        pattern_points = np.zeros((np.prod(self.pattern_size), 3),\
            np.float32)

        pattern_points[:, :2] = np.indices(self.pattern_size).T\
            .reshape(-1, 2)

        cv2.namedWindow("captureStream", cv2.CV_WINDOW_AUTOSIZE)
        cv2.namedWindow("patternDetection", cv2.CV_WINDOW_AUTOSIZE)

        save_files = utils.getAnswer("Would you like \
            to save picture files ? (y/n)  ", 'yn')

        finished_parsing = False

        if self.cam != '':
            while not finished_parsing and (self.max_frames_i == -1 or n_frames < self.max_frames_i):
                success, new_frame = self.cam.read()
                
                if success:
                    # Convert to B&W (if necessary ?)
                    grey_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
    
                    found, corners = cv2.findChessboardCorners(\
                        grey_frame, self.pattern_size)
    
                    cv2.imshow("captureStream", new_frame)
                    cv2.waitKey(2)
    
                    if found:
                        # Refine position
                        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT,
                                30, 0.1)
    
                        cv2.cornerSubPix(grey_frame, corners,
                                         (11, 11), (-1, -1), term)
    
                        # Draw detected pattern
                        cv2.drawChessboardCorners(new_frame,
                                                  self.pattern_size,
                                                  corners, found)
    
                        cv2.imshow("patternDetection", new_frame)
                        cv2.waitKey()
    
                        # Store values
                        self.img_points.append(corners.reshape(-1, 2))
                        self.obj_points.append(pattern_points)
    
                        n_frames += 1
                        print "{} patterns found".format(n_frames)
    
                        if save_files == 'y':
                            cv2.imwrite("calib_{}.bmp".format(n_frames),
                                        grey_frame)
    
                            cv2.imwrite("calib_{}_pattern.bmp_".format(n_frames),
                                        new_frame)
                
        self.frame_size = (len(new_frame[0]), len(new_frame))

        cv2.destroyAllWindows()

    def recordPattern_files(self):
        # Get patterns on every picture in "pictures[]"
        pattern_points = np.zeros((np.prod(self.pattern_size), 3),
                                  np.float32)

        pattern_points[:, :2] = np.indices(self.pattern_size).T.reshape(-1, 2)
        pattern_points[:, 0] = pattern_points[:, 0] * self.sq_size_h
        pattern_points[:, 1] = pattern_points[:, 1] * self.sq_size_v

        n_frames    = 0
        n_count     = 0
        b_left      = False
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
                term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT,
                        30, 0.1)
                cv2.cornerSubPix(new_frame, corners, (11, 11), (-1, -1), term)

                # Draw detected pattern
                cv2.drawChessboardCorners(new_frame,
                                          self.pattern_size,
                                          corners, found)
                                          
                # Resize
                if new_frame.shape[1]>self.frame_size_max[0]:
                    new_size = (int(new_frame.shape[0]/float(new_frame.shape[1]) * self.frame_size_max[0]), self.frame_size_max[0])
                                          
                else:
                    new_size = new_frame.shape
                    
                resized_pict = cv2.resize(new_frame,(new_size[1],new_size[0]))
                
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
                            b_skip_next = False # Should be useless
                            b_left = False

                        # Left picture
                        else:
                            self.img_points_l.append(corners.reshape(-1, 2))
                            self.obj_points_l.append(pattern_points)
                            b_skip_next = False # Don't skip next picture
                            b_left = True

                    n_frames += 1

                else :
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

                if n_frames ==1:
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
        print "If you calibrate from files, make sure stereo files are"
        print "in the left-right-left-right order"

        # Get settings & files
        self.stereo = True
        self.chooseCalibrationSettings()

        # Get patterns
        self.recordPatterns()

        # Compute the intrisic parameters first :
        rvecs = [np.zeros(3, dtype = np.float32) for i in xrange(self.max_frames_i)]
        tvecs = [np.zeros(3, dtype = np.float32) for i in xrange(self.max_frames_i)]

        # Call OpenCV routines to do the dirty work
        print "Computing intrisic parameters for the first camera"
        res = cv2.calibrateCamera(self.obj_points_l, self.img_points_l,
                                  self.frame_size,
                                  self.intrinsics_l, self.distorsion_l,
                                  rvecs, tvecs)

        rms, self.intrinsics_l, self.distorsion_l, rvecs, tvecs = res

        print "Computing intrisic parameters for the second camera"
        res = cv2.calibrateCamera(self.obj_points_r, self.img_points_r,
                                  self.frame_size,
                                  self.intrinsics_r, self.distorsion_r,
                                  rvecs, tvecs)
                                  
        rms, self.intrinsics_r, self.distorsion_r, rvecs, tvecs = res

        # Allocate arrays for the two camera matrix and distorsion matrices
        R = np.zeros((3, 3), dtype = np.float32)
        T = np.zeros((3, 1), dtype = np.float32)
        E = np.zeros((3, 3), dtype = np.float32)
        F = np.zeros((3, 3), dtype = np.float32)

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
#        stereo_flags |= cv2.CALIB_RATIONAL_MODEL          # Use 8 param rational distortion model instead of 5 param plumb bob model

        res = cv2.stereoCalibrate(self.obj_points_l,
                                  self.img_points_l, self.img_points_r,
                                  self.frame_size,
                                  self.intrinsics_l, self.distorsion_l,
                                  self.intrinsics_r, self.distorsion_r)

                                  # flags=stereo_flags

        (rms, int_left, dist_left, int_right, dist_right, R, T, E, F) = res

        # Output
        print "Calibration done"
        print "Residual RMS : {}".format(rms)

        if rms > 1.0:
            print "Calibration looks faulty, please re-run"

        print "\nCalibration parameters : \n Intrinsics -left \n {} \n\n Distorsion -left\n {}".format(\
            int_left, dist_left)

        print "\nCalibration parameters : \n Intrinsics -right \n {} \n\n Distorsion -right\n {}".format(\
            int_right, dist_right)

        print "\nRotation : \n{}\n".format(R)

        print "\nTranslation : \n{}\n".format(T)

        print "\nEssential matrix : \n{}\n".format(E) # Essential matrix

        print "\nFundamental matrix : \n{}\n".format(F) # Fundamental matrix

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
                        file = utils.handlePath(filepath, "camera_calibration.xml")

                        utils.saveParametersXML(int_left,
                                                dist_left,
                                                int_right,
                                                dist_right,
                                                R,
                                                T,
                                                self.frame_size,
                                                file)

                    else:
                        file_left = utils.handlePath(filepath, "_left.txt")
                        utils.saveParameters(int_left, dist_left, R, T, file_left, self.n_pattern_found)

                        file_right = utils.handlePath(filepath, "_right.txt")
                        utils.saveParameters(int_right, dist_right, R, T, file_right, self.n_pattern_found)

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
        rvecs = [np.zeros(3) for i in xrange(self.max_frames_i)]    # Rotation and translation matrix
        tvecs = [np.zeros(3) for i in xrange(self.max_frames_i)]

        _obj_points = np.array(self.obj_points, dtype=np.float32)
        _img_points = np.array(self.img_points, dtype=np.float32)

        rms, self.intrinsics, self.distorsion, _rvecs, _tvecs = cv2.calibrateCamera(_obj_points, _img_points, self.frame_size, self.intrinsics, self.distorsion, rvecs, tvecs)

        print "Calibration done"
        np.set_printoptions(precision=2)    
        np.set_printoptions(suppress=True)
        
        print "Residual RMS (pxl units) :\n"
        print(rms)

        print "\nRotations :\n"
        print(rvecs)

        print "\nTranslations :\n"
        print(tvecs)

        print "\nCalibration parameters :"
        print "Intrinsics \n"
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
                    filepath = utils.handlePath(filepath, "calib_results.txt")
    
                    # Create a file object in "write" mode
                    try:
                        utils.saveParameters(self.intrinsics, self.distorsion,
                            rvecs, tvecs, rms, filepath, self.n_pattern_found)

                        b_write_success = True
    
                    except:
                        print "Wrong path, please correct"
    
                    time.sleep(2)

        else:
            calib_file_path = utils.handlePath(self.file_path, "calib_results.txt")
            
            utils.saveParameters(self.intrinsics, self.distorsion,
                rvecs, tvecs, rms, calib_file_path, self.n_pattern_found)
            
            print "Saved calibration file"            
            
        return
