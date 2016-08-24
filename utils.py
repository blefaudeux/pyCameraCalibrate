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


def getAnswer(question, possibilities):
    answer = 'null'
    while not answer or possibilities.find(answer) == -1:
        answer = raw_input(question)

    return answer


def handlePath(path, filename):  # Handle paths and filenames.. return a correct pathway (hopefully)
    return os.path.join(path, filename)


def getCam():
    # Test the cams connected to the system :
    choice_done = False
    n_cam = 0
    cam = None

    while not choice_done:
        # start cam and show it :
        print "Capturing camera {} \n".format(n_cam)
        cam = cv2.VideoCapture(n_cam)

        if not cam:
            print "No more camera on the system"
            cam = ''
            break

        success, new_frame = cam.read()

        cv2.namedWindow("getCam", cv2.CV_WINDOW_AUTOSIZE)
        cv2.imshow("getCam", new_frame)
        cv2.waitKey(100)
        cv2.destroyWindow("getCam")

        answer = getAnswer("Is this the good camera ? (y/n)", 'yn')

        if answer == 'y':
            choice_done = True

        else:
            n_cam += 1

    return cam


def ndprint(a, format_string ='{0:.2f}'):   # http://stackoverflow.com/questions/2891790/pretty-printing-of-numpy-array
    print [format_string.format(v, i) for i, v in enumerate(a)]


def showCam(cam_number):
    cam = cv2.VideoCapture(cam_number)
    key = -1

    cv2.namedWindow("showCam")

    while key < 0:
        success, img = cam.read()
        cv2.imshow("showCam", img)
        key = cv2.waitKey(1)

    cv2.destroyWindow("showCam")
    time.sleep(2)
    print "Leaving showCam"


def sortNicely(l):
    # Sort the given list in the way that humans expect.
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )
    return l

