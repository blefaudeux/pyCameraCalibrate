# -*- coding: utf-8 -*-

#!/usr/bin/env python

"""
Created on Wed Oct 31 14:26:31 2012

@author: Benjamin Lefaudeux (blefaudeux at github)

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

import cv2
import time
import os
import re


# Import XML
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


def getAnswer(question, possibilities):
    answer = 'null'
    while (possibilities.find(answer) == -1):
        answer = raw_input(question)
    return answer

# Handle paths and filenames.. return a correct pathway (hopefully)
def handlePath(path, filename):
    if len(path) == 0:
        return filename

    else:
        if (path[-1] != '/') :
            # Test if its a directory
            if (os.path.isdir(path)):
                new_path = path + "/" + filename
                return new_path

            # Test if its a filename
            else :
                if (os.path.isfile(path)):
                    return path

        # It's just the beginning of a path, complete
        else :
            new_path = path + filename
            return new_path

def getCam():
    # Test the cams connected to the system :
    choice_done = False
    n_cam = 0

    while (not(choice_done)):
        # start cam and show it :
        print "Capturing camera {} \n".format(n_cam)
        cam = cv2.VideoCapture(n_cam)

        if not cam:
            print "No more camera on the system"
            choice_done = True
            cam = ''
            break

        success, new_frame = cam.read()

        cv2.namedWindow("getCam", cv2.CV_WINDOW_AUTOSIZE)
        cv2.imshow("getCam", new_frame)
        cv2.waitKey(100)
        cv2.destroyWindow("getCam")

        answer = getAnswer("Is this the good camera ? (y/n)", 'yn')

        if (answer == 'y'):
            choice_done = True

        else:
            n_cam = n_cam + 1

    return cam

# http://stackoverflow.com/questions/2891790/pretty-printing-of-numpy-array
def ndprint(a, format_string ='{0:.2f}'):
    print [format_string.format(v,i) for i,v in enumerate(a)]

def saveParameters(intrinsic, distorsion, rotation, translation, rms, path):
    FILE = open(path, "w")

    # Write parameters :
    FILE.write("Calibration error (pixels) : \n")
    FILE.write("{}\n\n".format(rms))

    FILE.write("Intrisic Matrix : \n")
    FILE.write("{}\n\n".format(intrinsic))

    FILE.write("Distorsion coefficients :\n")
    FILE.write("{}\n\n".format(distorsion))

    FILE.write("Rotations :\n")
    FILE.write("{}\n\n".format(rotation))

    FILE.write("Translations :\n")
    FILE.write("{}\n\n".format(translation))

    # Close file
    FILE.close()

def saveParametersXML(intrinsic_0,
                      distorsion_0,
                      intrinsic_1,
                      distorsion_1,
                      rotation,
                      translation,
                      picture_size,
                      path):

    # Build XML structure from the settings to be saved
    cam_calibration = ET.Element('camera_calibration')
    cam_mat_0 = ET.SubElement(cam_calibration, 'camera_matrix_0')
    cam_mat_1 = ET.SubElement(cam_calibration, 'camera_matrix_1')
    dist_mat_0 = ET.SubElement(cam_calibration, 'dist_matrix_0')
    dist_mat_1 = ET.SubElement(cam_calibration, 'dist_matrix_1')
    rot_mat = ET.SubElement(cam_calibration, 'rotation_matrix')
    tr_mat = ET.SubElement(cam_calibration, 'translation_matrix')
    pict_size = ET.SubElement(cam_calibration, 'picture_size')

    # Fill in first camera matrix
    cam_mat_0.set('a0', str(intrinsic_0[0,0]))
    cam_mat_0.set('a1', str(intrinsic_0[0,1]))
    cam_mat_0.set('a2', str(intrinsic_0[0,2]))
    cam_mat_0.set('b0', str(intrinsic_0[1,0]))
    cam_mat_0.set('b1', str(intrinsic_0[1,1]))
    cam_mat_0.set('b2', str(intrinsic_0[1,2]))
    cam_mat_0.set('c0', str(intrinsic_0[2,0]))
    cam_mat_0.set('c1', str(intrinsic_0[2,1]))
    cam_mat_0.set('c2', str(intrinsic_0[2,2]))

    # Fill in second camera matrix
    cam_mat_1.set('a0',str(intrinsic_1[0,0]))
    cam_mat_1.set('a1',str(intrinsic_1[0,1]))
    cam_mat_1.set('a2',str(intrinsic_1[0,2]))
    cam_mat_1.set('b0',str(intrinsic_1[1,0]))
    cam_mat_1.set('b1',str(intrinsic_1[1,1]))
    cam_mat_1.set('b2',str(intrinsic_1[1,2]))
    cam_mat_1.set('c0',str(intrinsic_1[2,0]))
    cam_mat_1.set('c1',str(intrinsic_1[2,1]))
    cam_mat_1.set('c2',str(intrinsic_1[2,2]))

    # Fill in first distorsion matrix
    dist_mat_0.set('k1', str(distorsion_0[0]))
    dist_mat_0.set('k2', str(distorsion_0[1]))
    dist_mat_0.set('p1', str(distorsion_0[2]))
    dist_mat_0.set('p2', str(distorsion_0[3]))
    dist_mat_0.set('k3', str(distorsion_0[4]))
#    dist_mat_0.set('k4', str(distorsion_0[0,5]))
#    dist_mat_0.set('k5', str(distorsion_0[0,6]))
#    dist_mat_0.set('k6', str(distorsion_0[0,7]))

    # Fill in second distorsion matrix
    dist_mat_1.set('k1', str(distorsion_1[0]))
    dist_mat_1.set('k2', str(distorsion_1[1]))
    dist_mat_1.set('p1', str(distorsion_1[2]))
    dist_mat_1.set('p2', str(distorsion_1[3]))
    dist_mat_1.set('k3', str(distorsion_1[4]))
#    dist_mat_1.set('k4', str(distorsion_1[0,5]))
#    dist_mat_1.set('k5', str(distorsion_1[0,6]))
#    dist_mat_1.set('k6', str(distorsion_1[0,7]))

    # Fill in rotation matrix
    rot_mat.set('a0', str(rotation[0,0]))
    rot_mat.set('a1', str(rotation[0,1]))
    rot_mat.set('a2', str(rotation[0,2]))
    rot_mat.set('b0', str(rotation[1,0]))
    rot_mat.set('b1', str(rotation[1,1]))
    rot_mat.set('b2', str(rotation[1,2]))
    rot_mat.set('c0', str(rotation[2,0]))
    rot_mat.set('c1', str(rotation[2,1]))
    rot_mat.set('c2', str(rotation[2,2]))

    # Fill in translation matrix
    tr_mat.set('a0', str(translation[0,0]))
    tr_mat.set('a1', str(translation[1,0]))
    tr_mat.set('a2', str(translation[2,0]))

    # Fill in picture size
    pict_size.set('width', str(picture_size[0]))
    pict_size.set('height', str(picture_size[1]))

    # Write to file
    tree = ET.ElementTree(cam_calibration)
    tree.write(path)


def showCam(cam_number):
    cam = cv2.VideoCapture(cam_number)
    key = -1

    cv2.namedWindow("showCam")

    while(key < 0):
        success, img = cam.read()
        cv2.imshow("showCam", img)
        key = cv2.waitKey(1)

    cv2.destroyWindow("showCam")
    time.sleep(2)
    print "Leaving showCam"

def sortNicely(l ):
      """ Sort the given list in the way that humans expect.
      """
      convert = lambda text: int(text) if text.isdigit() else text
      alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
      l.sort( key=alphanum_key )
      return l

