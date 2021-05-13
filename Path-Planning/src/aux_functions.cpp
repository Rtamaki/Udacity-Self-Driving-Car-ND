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
  vector<vector<double>> new_sensor_fusion,
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

      safe_distance = ref_velocity * safety_factor;


      ptsx.clear();
      ptsy.clear();
      sensor_fusion = new_sensor_fusion;

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
      vector<double> next_wp0 = getXY(ref_s+distance_horizon, (2+4*lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
      vector<double> next_wp1 = getXY(ref_s+2*distance_horizon, (2+4*lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
      vector<double> next_wp2 = getXY(ref_s+3*distance_horizon, (2+4*lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);

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




// This functions detects if there is a car too close ahead to the car in the same lane. If there is, the car slows down.
void PathPlanner::CarTooClose(){

  too_close_car = false;
  for(int i = 0; i < sensor_fusion.size(); i++)
  {

    // Verify if the car is in the same lane.
    float d = sensor_fusion[i][6];
    if(d > 4*lane - car_half_width && d < 4*lane + 4 + car_half_width)
    {
      // If the car is in the same lane, I have to check if we are too close to it (s coordinate)
      // and if it is going slower than us
      double vx = sensor_fusion[i][3];
      double vy = sensor_fusion[i][4];
      double check_speed = sqrt(vx*vx + vy*vy);
      double check_car_s = sensor_fusion[i][5];
      double check_car_s_plus = check_car_s + time_increment*check_speed; // car next position
      if((check_car_s > ref_s || check_car_s_plus > ref_s) && (check_car_s - ref_s < safe_distance || check_car_s_plus - ref_s < safe_distance) && check_speed < ref_velocity)
      {
        too_close_car = true;
      }
    }
  }
}



void PathPlanner::ControlVelocity(){

  if(too_close_car)
  {
    ref_velocity-= speed_increment;
  }else{
    if(ref_velocity + 2*speed_increment < limit_velocity)
    {
      ref_velocity+= speed_increment;
    }
  }
}


// Function only for debug
void PathPlanner::CheckValues(){

    cout << "Too close? " << too_close_car << endl;
    cout << "ref_velocity" << ref_velocity << endl;
}



// This function will be added to the path planning function
// It will check out the other lanes for a possible change lane, if there is a
// lane where the car CAN move to without crashing ad if there are no cars
// immediately ahead or if the other cars are going faster, than it will change the car lane/
// As an output, it will say if we are goingto change lanes, so that we don't need
// to slow down
void PathPlanner::Try2ChangeLanes(){


  // Vector that will contain the position of the car that is closest to us in
  // in a lane. This will also consider cars a bit behind us, so that they
  // don't collide on us when we change lanes.
  vector<double> closet_car_dist(number_of_lanes);
  closet_car_dist.assign(number_of_lanes, 5*safe_distance);

  // Only need to run if we sense there is a car too close to us, otherwise
  // we keep in the same lane
  if(too_close_car)
  {
    // For all the sensed cars
    for(int i = 0; i < sensor_fusion.size(); i++)
    {

      // Attribute the car to a lane
      int sensed_car_lane = (int)round((sensor_fusion[i][6] - 2.0)/4);
      double vx = sensor_fusion[i][3];
      double vy = sensor_fusion[i][4];
      double check_speed = sqrt(vx*vx + vy*vy);
      double check_car_s = sensor_fusion[i][5];

      // check if the car is in  different lane, a lane whre the car can move
      if(sensed_car_lane != lane && abs(sensed_car_lane - lane) == 1)
      {
          // Check if the sensed car IS or WILL BE too close in the future
          if(check_car_s > ref_s - safe_distance*dist_lb_fraction && check_car_s - ref_s< closet_car_dist[sensed_car_lane])
          {
            closet_car_dist[sensed_car_lane] = check_car_s - ref_s;
          }

          double rel_velocity = check_speed - ref_velocity;
          if(check_car_s + rel_velocity*time_increment > ref_s - safe_distance*dist_lb_fraction && check_car_s + rel_velocity*time_increment - ref_s < closet_car_dist[sensed_car_lane])
          {
            closet_car_dist[sensed_car_lane] = check_car_s + rel_velocity*time_increment - ref_s;
          }
      }
    }


    // After checking out all the cars and saving the information of the closest
    // car to us in each of the valid lanes, I will look from left to right if
    // there is at least one lane where we can move safely

    for(int i = 0; i < number_of_lanes; i++)
    {
      // Current lane is not valid, obviously
      if(i != lane && abs(i - lane) == 1)
      {
        if((closet_car_dist[i]) > safe_distance)
        {
          lane = i;
          // If the car can change lanes, then the car in front of us is not actually too close
          too_close_car = false;
          break;
        }

      }

    }
  }





}
