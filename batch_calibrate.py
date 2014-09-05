# -*- coding: utf-8 -*-

#!/usr/bin/env python

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

import camera_calibration as cam_calib
import utils
import os

class batchCalibration:
    def __init__(self):
        self.root_path      = ''
        self.folder_list    = []
        self.pattern_size   = (0,0) # Number of inner squares on each dimension
        self.sq_size_h      = 0.0
        self.sq_size_v      = 0.0
    
    def calibrate(self, usedefaults = True):
        
        if not usedefaults:
            # Get the pattern parameters
            h_dim = utils.getAnswer(
            "Number of inner corners on the horizontal dimension ? ", '12345678910')
    
            v_dim = utils.getAnswer(
            "Number of inner corners on the vertical dimension ? ", '12345678910')
    
            # Enter the number of squares over each dimensions
            self.pattern_size = (int(h_dim), int(v_dim))
            print "Chessboard dimensions : {} x {}"\
                .format(self.pattern_size[0], self.pattern_size[1])
        
            get_square_size = False
            while not(get_square_size):
                sq_size = raw_input("Horizontal Size (m) of the squares ? ")
    
                try:
                    self.sq_size_h = float(sq_size)
                    get_square_size = True
    
                except ValueError:
                    print "Cannot determine dimension"
    
            get_square_size = False
            while not(get_square_size):
                sq_size = raw_input("Vertical Size (m) of the squares ? ")
    
                try:
                    self.sq_size_v = float(sq_size)
                    get_square_size = True
    
                except ValueError:
                    print "Cannot determine dimension"    
                    
        else :
            h_dim = 9
            v_dim = 6
            self.pattern_size = (int(h_dim), int(v_dim))                
            self.sq_size_h = 0.02545
            self.sq_size_v = 0.02545
            
            print "Used parameters :"
            print "Pattern size : {}".format(self.pattern_size)
            print "Physical dimensions : {}m x {}m \n ".format(self.sq_size_h, self.sq_size_v)
            
        # Get the root folder, Get all the subfolders, 
        # do all the subsequent calibrations and record the results 
        path = raw_input("Root path for the calibration folders : ")           
        
        path = os.path.join(path, '')
       
        for dirpath, dirnames, filenames in os.walk(path):

            for dir in dirnames :
                if not dir[0] == '.':
                    path = dirpath + dir
                    
                    settings = (False, True, True, True, path, self.pattern_size , 
                                (self.sq_size_h,self.sq_size_v), False)                
                    
                    new_cam = cam_calib.cameraCalibration(settings)
                    
                    print "\nCalibrating using files in folder : {}".format(dir)                
                    
                    if (os.path.exists(path +'/calib_results.txt')):
                        print "Folder {} already contains calibration results".format(dir)
                    else :
                        new_cam.calibrate()
                        
        raw_input("\nCalibration done, press any key to exit")


# Run this script
new_run = batchCalibration()
new_run.calibrate( )