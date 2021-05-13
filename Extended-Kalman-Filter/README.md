# Extended Kalman Filter Project Starter Code
Self-Driving Car Engineer Nanodegree Program

I implemented a Extended Kalman Filter algorithm to predict and estimate the position of a moving object(in this case a vehicle) from data measured from LIDAR and RADAR sensors connected to this vehicle.
The algorithm is called Extended becausewe consider the measurements taken not only from one, but from 2 sensors types: LIDAR, which provides information about the position, and RADAR, which gives information about the velocities. 
When using these two sensors types, we need to define a coordinate system to be used for all the project and convert the data to this coordinate system if necessary. In this case, we utilized the cartesian coordinate system. The data from LIDAR comes in this coordinate sytem, however, those from RADAR come in polar coordinate system, therefore a conversion is necessary. WIth this exception considered, the rest is just basic Kalman Filter.

In the data, we assumed the convariance matrix Q(of the prediction step) to be 9 for both axis X and Y. Additionally, one can change the initial value for the process covariance matrix P. If it is too large, a lot of the will be necessary until the convariance reaches the covariance of matrix Q and the RMSE can be large. If we choose a small initial value for P, less time will be required until it reaches the convariance Q, but if we arbitrarily change it, we ar only aritificially improving the RMSE, so it is not advisable. 

Since this is only a simple Kalman filter, there aren't many parameters to adjust to achieve better results. What is possible to improve the precision lies more in the hardware part, such as improving the precision of the sensors, increasing the frequency of the measurements, and lowerin the variance in the measurements. However, were we to use a non linear estimation for the Kalman Filter, than we could adjust parameters for performance improvement.

There are some points for improvements:
1) Increasethe measurements frequency to lower the uncertainty in the prediction step
2) Increase measurement precision to lower variance 
3) Use a nonlinear estimation for the Kalman FIlter to make it more robust against curves



# To compile and run this project
Install uWebSocketIO. Then go to the 'build' directory and compile by writing "cmake ..&& make" and then "./ExtendedKF".
