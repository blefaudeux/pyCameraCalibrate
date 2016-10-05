#!/usr/bin/python2

# -*- coding: utf-8 -*-

"""
Created on Mon Nov 11 11:32:37 2013

@author: Benjamin Lefaudeux (blefaudeux at github)

This script uses OpenCV to calibrate a batch of cameras in one run.

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

import argparse
import camera_calibration as cam_calib
import utils
import os


def calibrate(settings=None):

    if not settings:
        settings = cam_calib.CameraCalibrationSettings()

        # Get the pattern parameters
        h_dim = utils.getAnswer("Number of inner corners on the horizontal dimension ? ", '12345678910')
        v_dim = utils.getAnswer("Number of inner corners on the vertical dimension ? ", '12345678910')

        # Enter the number of squares over each dimensions
        settings.pattern_size = (int(h_dim), int(v_dim))
        print("Chessboard dimensions : {} x {}"
              .format(settings.pattern_size[0], settings.pattern_size[1]))

        get_square_size = False
        while not get_square_size:
            sq_size = raw_input("Horizontal Size (m) of the squares ? ")

            try:
                settings.sq_size_h = float(sq_size)
                get_square_size = True

            except ValueError:
                print("Cannot determine dimension")

        get_square_size = False
        while not get_square_size:
            sq_size = raw_input("Vertical Size (m) of the squares ? ")

            try:
                settings.sq_size_v = float(sq_size)
                get_square_size = True

            except ValueError:
                print("Cannot determine dimension")

    else:
        print("---\nUsed parameters")
        print("Pattern size : {}".format(settings.pattern_size))
        print("Physical dimensions : {}m x {}m \n---\n ".format(settings.sq_size_h, settings.sq_size_v))

    # Get the root folder, Get all the subfolders, do all the subsequent calibrations and record the results
    if len(settings.file_path) == 0:
        settings.file_path = os.path.join(raw_input("Root path for the calibration folders : "), '')

    for dirpath, dirnames, filenames in os.walk(settings.file_path):
        for directory in dirnames:
            if not directory[0] == '.' and directory != 'undistorted' :
                settings.file_path = os.path.join(dirpath, directory)
                new_cam = cam_calib.CameraCalibration(settings)

                print("Calibrating using files in folder : {}".format(directory))

                if os.path.exists(os.path.join(settings.file_path, "calib_results.json")):
                    print("Folder {} already contains calibration results".format(directory))
                else:
                    new_cam.calibrate()

    raw_input("\nCalibration done, press any key to exit")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calibrate camera(s)')

    parser.add_argument(
        '-st', '--stereo', dest='stereo', action='store',
        help='Calibrate stereocameras',
        default=False
    )

    parser.add_argument(
        '-i', '--interactive', dest='interactive', action='store',
        help='Validate pattern detection interactively',
        default=False
    )

    parser.add_argument(
        '-d', '--data_path', dest='data_path', action='store',
        help='Root path for the data',
        default=""
    )

    parser.add_argument(
        '-s', '--save_results', dest='save', action='store',
        help='Save calibration results',
        default=True
    )

    parser.add_argument(
        '-m', '--max_patterns', dest='max_patterns', action='store',
        help='Max number of patterns to be used',
        default=20
    )

    parser.add_argument(
        '-sh', '--size_h', dest='size_horizontal', action='store',
        help='Horizontal size of the pattern (meters)',
        default=0.003
    )

    parser.add_argument(
        '-sv', '--size_v', dest='size_vertical', action='store',
        help='Vertical size of the pattern (meters)',
        default=0.003
    )

    parser.add_argument(
        '-nh', '--number_h', dest='number_horizontal', action='store',
        help='Horizontal number of inner corners',
        default=8
    )

    parser.add_argument(
        '-nv', '--number_v', dest='number_vertical', action='store',
        help='Vertical number of inner corners',
        default=6
    )

    parser.add_argument(
        '-f', '--focal-guess', dest='focal', action='store',
        help='Initial guess for the focal length',
        default=-1
    )

    parser.add_argument(
        '-c', '--use_live_camera', dest='use_camera', action='store_true',
        help='Use a live camera stream'
    )

    args = parser.parse_args()
    settings = cam_calib.CameraCalibrationSettings()
    settings.auto_save = args.save
    settings.auto_validation = not args.interactive
    settings.pattern_size = (int(args.number_horizontal), int(args.number_vertical))
    settings.sq_size_h = args.size_horizontal
    settings.sq_size_v = args.size_vertical
    settings.file_path = args.data_path
    settings.focal_guess = args.focal

    calibrate(settings)
