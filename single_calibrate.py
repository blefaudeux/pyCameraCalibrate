# -*- coding: utf-8 -*-

#!/usr/bin/python

"""
Created on Mon Nov 11 11:32:37 2013

@author: Benjamin Lefaudeux (blefaudeux at github)

This script uses OpenCV to calibrate a single camera.

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

import camera_calibration as cam_utils

# Run script
new_instance = cam_utils.cameraCalibration()
new_instance.calibrate()
