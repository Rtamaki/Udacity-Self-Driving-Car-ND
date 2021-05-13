# include "aux_functions.h"

using namespace std;




constexpr double pi();
double deg2rad(double x);
double rad2deg(double x);

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_first_of("}");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

/*
    Moved the function from main.cpp to here, so that the functions from path_planner could you them as well
*/


// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.


double distance(double x1, double y1, double x2, double y2)
{
	return sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
}
int ClosestWaypoint(double x, double y, const vector<double> &maps_x, const vector<double> &maps_y)
{

	double closestLen = 100000; //large number
	int closestWaypoint = 0;

	for(int i = 0; i < maps_x.size(); i++)
	{
		double map_x = maps_x[i];
		double map_y = maps_y[i];
		double dist = distance(x,y,map_x,map_y);
		if(dist < closestLen)
		{
			closestLen = dist;
			closestWaypoint = i;
		}

	}

	return closestWaypoint;

}

int NextWaypoint(double x, double y, double theta, const vector<double> &maps_x, const vector<double> &maps_y)
{

	int closestWaypoint = ClosestWaypoint(x,y,maps_x,maps_y);

	double map_x = maps_x[closestWaypoint];
	double map_y = maps_y[closestWaypoint];

	double heading = atan2((map_y-y),(map_x-x));

	double angle = fabs(theta-heading);
  angle = min(2*pi() - angle, angle);

  if(angle > pi()/4)
  {
    closestWaypoint++;
  if (closestWaypoint == maps_x.size())
  {
    closestWaypoint = 0;
  }
  }

  return closestWaypoint;
}

// Transform from Cartesian x,y coordinates to Frenet s,d coordinates
vector<double> getFrenet(double x, double y, double theta, const vector<double> &maps_x, const vector<double> &maps_y)
{
	int next_wp = NextWaypoint(x,y, theta, maps_x,maps_y);

	int prev_wp;
	prev_wp = next_wp-1;
	if(next_wp == 0)
	{
		prev_wp  = maps_x.size()-1;
	}

	double n_x = maps_x[next_wp]-maps_x[prev_wp];
	double n_y = maps_y[next_wp]-maps_y[prev_wp];
	double x_x = x - maps_x[prev_wp];
	double x_y = y - maps_y[prev_wp];

	// find the projection of x onto n
	double proj_norm = (x_x*n_x+x_y*n_y)/(n_x*n_x+n_y*n_y);
	double proj_x = proj_norm*n_x;
	double proj_y = proj_norm*n_y;

	double frenet_d = distance(x_x,x_y,proj_x,proj_y);

	//see if d value is positive or negative by comparing it to a center point

	double center_x = 1000-maps_x[prev_wp];
	double center_y = 2000-maps_y[prev_wp];
	double centerToPos = distance(center_x,center_y,x_x,x_y);
	double centerToRef = distance(center_x,center_y,proj_x,proj_y);

	if(centerToPos <= centerToRef)
	{
		frenet_d *= -1;
	}

	// calculate s value
	double frenet_s = 0;
	for(int i = 0; i < prev_wp; i++)
	{
		frenet_s += distance(maps_x[i],maps_y[i],maps_x[i+1],maps_y[i+1]);
	}

	frenet_s += distance(0,0,proj_x,proj_y);

	return {frenet_s,frenet_d};

}

// Transform from Frenet s,d coordinates to Cartesian x,y
vector<double> getXY(double s, double d, const vector<double> &maps_s, const vector<double> &maps_x, const vector<double> &maps_y)
{
	int prev_wp = -1;

	while(s > maps_s[prev_wp+1] && (prev_wp < (int)(maps_s.size()-1) ))
	{
		prev_wp++;
	}

	int wp2 = (prev_wp+1)%maps_x.size();

	double heading = atan2((maps_y[wp2]-maps_y[prev_wp]),(maps_x[wp2]-maps_x[prev_wp]));
	// the x,y,s along the segment
	double seg_s = (s-maps_s[prev_wp]);

	double seg_x = maps_x[prev_wp]+seg_s*cos(heading);
	double seg_y = maps_y[prev_wp]+seg_s*sin(heading);

	double perp_heading = heading-pi()/2;

	double x = seg_x + d*cos(perp_heading);
	double y = seg_y + d*sin(perp_heading);

	return {x,y};

}

















/*
  Functions for the class Path Planner
*/

PathPlanner::PathPlanner() {};
PathPlanner::~PathPlanner() {};

vector<double> PathPlanner::transform2car_coord(double world_coord_x, double world_coord_y){
  // This function transform the points to the car coordinate system
  double shift_x = world_coord_x - ref_x;
  double shift_y = world_coord_y - ref_y;

  double card_coord_x = (shift_x*cos(0-ref_yaw) - shift_y*sin(0-ref_yaw));
  double card_coord_y = (shift_x*sin(0-ref_yaw) + shift_y*cos(0-ref_yaw));

  return vector<double> {card_coord_x, card_coord_y};
}


vector<double> PathPlanner::transform2world_coord(double car_coord_x, double car_coord_y){
  // This function transform the points to the car coordinate system

  double world_coord_x = (car_coord_x*cos(ref_yaw) - car_coord_y*sin(ref_yaw)) + ref_x;
  double world_coord_y = (car_coord_x*sin(ref_yaw) + car_coord_y*cos(ref_yaw)) + ref_y;

  return vector<double> {world_coord_x, world_coord_y};

}



//Function to update the values of the path planner class for each time stamp
void PathPlanner::Update(double car_x,
  double car_y,
  double car_yaw,
  double car_s,
  double car_d,
  vector<double> previous_path_x,
  vector<double> previous_path_y,
  vector<double> map_waypoints_x,
  vector<double> map_waypoints_y,
  vector<double> map_waypoints_s)
  {
      ref_x = car_x;
      ref_y = car_y;
      ref_yaw = car_yaw;
      ref_s = car_s;
      ref_d = car_d;
      previous_ptsx = previous_path_x;
      previous_ptsy = previous_path_y;
      int prev_size = previous_path_x.size();

      if(prev_size < 2)
      {
        double prev_car_x = ref_x - cos(ref_yaw);
        double prev_car_y = ref_y - sin(ref_yaw);

        ptsx.push_back(prev_car_x);
        ptsy.push_back(prev_car_y);

        ptsx.push_back(ref_x);
        ptsy.push_back(ref_y);
      }
      else{
          ref_x = previous_path_x[prev_size-1];
          ref_y = previous_path_y[prev_size-1];


          double ref_x_prev = previous_path_x[prev_size-2];
          double ref_y_prev = previous_path_y[prev_size-2];
          ref_yaw = atan2(ref_y - ref_y_prev, ref_x - ref_x_prev);

          ptsx.push_back(ref_x_prev);
          ptsy.push_back(ref_y_prev);

          ptsx.push_back(ref_x);
          ptsy.push_back(ref_y);
      }


      // Add some points in the horizon of 30m, 60m, and 90m ahead. Heere we are calling them waypoints
      vector<double> next_wp0 = getXY(ref_s+30, (2+4*lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
      vector<double> next_wp1 = getXY(ref_s+60, (2+4*lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
      vector<double> next_wp2 = getXY(ref_s+90, (2+4*lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);

      ptsx.push_back(next_wp0[0]);
      ptsx.push_back(next_wp1[0]);
      ptsx.push_back(next_wp2[0]);

      ptsy.push_back(next_wp0[1]);
      ptsy.push_back(next_wp1[1]);
      ptsy.push_back(next_wp2[1]);

  }



// Function that will plan the car trajectory based on the updated values
vector<vector<double>> PathPlanner::PlanTrajectory(){


  // transform from global coordinate system to car coordinate system
  vector<double> card_coord_x;
  vector<double> card_coord_y;
  for(int i=0; i<ptsx.size(); i++)
  {
    vector<double> car_xy = transform2car_coord(ptsx[i], ptsy[i]);
    card_coord_x.push_back(car_xy[0]);
    card_coord_y.push_back(car_xy[1]);
  }


  // Use spline library to generate the next point in the car coordinate system
  tk::spline s;
  s.set_points(card_coord_x, card_coord_y);

  // define the vector that will contain the next values
  vector<double> next_ptsx;
  vector<double> next_ptsy;
  // Fill the next points with all the previous points that werent used
  for(int i = 0; i < previous_ptsx.size(); i++)
  {
    next_ptsx.push_back(previous_ptsx[i]);
    next_ptsy.push_back(previous_ptsy[i]);
  }



  // Generate the next points with spline
  double target_x = distance_horizon;
  double target_y = s(target_x);
  double target_distance = sqrt(target_x*target_x + target_y*target_y);
  double add_on_x = 0; //Initially, I start at the car origin, but it will increment afterwards


  // Fill the rest of the next points with the new points from spline after having filled them with the previous point
  for(int i=1; i < points_amount - previous_ptsx.size(); i++)
  {
    double N = target_distance/(time_increment*ref_velocity);
    double x_point = add_on_x + target_x/N;
    double y_point = s(x_point);

    //update the current x coordinate
    add_on_x = x_point;


    vector<double> new_world_xy = transform2world_coord(x_point, y_point);
    next_ptsx.push_back(new_world_xy[0]);
    next_ptsy.push_back(new_world_xy[1]);
  }


  return vector<vector<double>> {next_ptsx, next_ptsy};






}
