#ifndef AUX_FUNCTION
#define AUX_FUNCTION

#include <fstream>
#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "json.hpp"
#include "spline.h"


using namespace std;

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s);

/*
    Moved the function from main.cpp to here, so that the functions from path_planner could you them as well
*/


// For converting back and forth between radians and degrees.
constexpr double pi();
double deg2rad(double x);
double rad2deg(double x);

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.


double distance(double x1, double y1, double x2, double y2);

int ClosestWaypoint(double x, double y, const vector<double> &maps_x, const vector<double> &maps_y);

int NextWaypoint(double x, double y, double theta, const vector<double> &maps_x, const vector<double> &maps_y);

// Transform from Cartesian x,y coordinates to Frenet s,d coordinates
vector<double> getFrenet(double x, double y, double theta, const vector<double> &maps_x, const vector<double> &maps_y);

// Transform from Frenet s,d coordinates to Cartesian x,y
vector<double> getXY(double s, double d, const vector<double> &maps_s, const vector<double> &maps_x, const vector<double> &maps_y);

























// Class that will run the algorithm to plan the path, so that main.cpp isn't overloaded with code
class PathPlanner{

  public:
    /*
    * Constructor
    */
    PathPlanner();

    /*
    * Destructor.
    */
    virtual ~PathPlanner();

    /*
    * Initializer.
    */
    void Init(double Kp, double Ki, double Kd);

    void Update(double car_x,
      double car_y,
      double car_yaw,
      double car_s,
      double car_d,
      vector<vector<double>> new_sensor_fusion,
      vector<double> previous_path_x,
      vector<double> previous_path_y,
      vector<double> map_waypoints_x,
      vector<double> map_waypoints_y,
      vector<double> map_waypoints_s);

      vector<vector<double>> PlanTrajectory();

      vector<double> transform2car_coord(double world_coord_x,
        double world_coord_y);

      vector<double> transform2world_coord(double car_coord_x,
         double car_coord_y);

       void CarTooClose();

       void ControlVelocity();

       void CheckValues();

       void Try2ChangeLanes();

  private:

    // Values containig the car current state
    int lane = 1;
    double rev_vel;
    double ref_x;
    double ref_y;
    double ref_yaw;
    double ref_s;
    double ref_d;
    vector<vector<double>> sensor_fusion;

    // Reference values
    const double distance_horizon = 50.0;
    const int points_amount = 50;
    double ref_velocity = 0/2.24;
    const double time_increment = 0.02;
    const double limit_velocity = 49/2.24;

    // Variables for lane change
    int number_of_lanes = 3;
    double dist_lb_fraction = 0.1;


    // Controls for the car not to crash
    float car_half_width = 1/2;
    double safe_distance = 30.0;
    const double safety_factor = 1.5;
    const double speed_increment = 0.2;
    bool too_close_car = false;

    vector<double> ptsx;
    vector<double> ptsy;
    vector<double> previous_ptsx;
    vector<double> previous_ptsy;

};








#endif /* PID_H */
