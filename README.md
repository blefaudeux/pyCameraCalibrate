pyCameraCalibrate
=================
- Calibrate a camera using python and opencv. 
- Calibrate a stereo camera setup.
- Calibrate a live camera, or using prior captures
- Calibrate batchs of cameras by pointing to the root folder structure containing pattern pictures, and generate a report.

Nothing fancy, just a useful tool in some circumstances..

# Usecase : 
## Batch calibration 
### Basic run : 
`python batch_calibrate.py` (some settings are defined by default, modify the script as needed)

### Additional optional parameters: 
- `-st, --stereo`: Calibrate stereo cameras
- `-i, --interactive`: Interactive pattern search (reject/validate on the fly)
- `-d, --data_path`: Where the files are stored
- `-s, --save_results`: Save the results in a JSON file
- `-m, --max_patterns`: Limit the maximum number of patterns used
- `-sh, --size_h`: Horizontal size of a pattern unit rectangle
- `-sv, --size_v`: Vertical size of a pattern unit rectangle
- `-sh, --size_h`: Horizontal size of a pattern unit rectangle
- `-nh, --number_h`: Number of inner corners on the horizontal axis
- `-nv, --number_v`: Number of inner corners on the vertical axis
- `-f, --focal_guess`: Input an initial value for the focal length


## One time calibration 
`python camera_calibration.py`

# Installation requisites :
* Python 2.xx (opencv bindings are a limiting factor here)
* OpenCV with python bindings