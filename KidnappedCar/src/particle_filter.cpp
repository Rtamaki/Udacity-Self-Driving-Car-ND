/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <map>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.

	// Define the number of particles
	if (is_initialized) {
	    return;
	  }
	num_particles = 100;

	// This line creates a normal (Gaussian) distribution for x, y and theta with mean in the GPS estimated position
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	// Create all the particles, add noise to the position and orientation values
	default_random_engine gen;
	for(int i = 0; i < num_particles; i++)
	{
		Particle random_particle;
		random_particle.id = i;
		random_particle.x = dist_x(gen);
		random_particle.y = dist_y(gen);
		random_particle.theta = dist_theta(gen);
		random_particle.weight = 1;
		particles.push_back(random_particle);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// Generate random engine and gaussian distribution to add noise to the predictions
	default_random_engine gen;
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

	for(int i = 0; i < num_particles; i++)
	{

		if(abs(yaw_rate) > 0.01)
		{
			particles[i].x += velocity/yaw_rate*(sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
			particles[i].y += velocity/yaw_rate*(-cos(particles[i].theta + yaw_rate*delta_t) + cos(particles[i].theta));
		}else{
			particles[i].x += velocity*cos(particles[i].theta)*delta_t;
			particles[i].y += velocity*sin(particles[i].theta)*delta_t;
		}

		particles[i].theta += yaw_rate*delta_t + dist_theta(gen);


		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);

	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

	int number_predicted_landmarks = predicted.size();

	// Variables to store the minimum distance found for each observed landmark
	double min_distance, distance;
	int closest_observation = -1;

	for(int i = 0; i < observations.size(); i++)
	{
		min_distance = std::numeric_limits<double>::max();
		for(int j = 0; j < number_predicted_landmarks; j++)
		{
			distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
			if(distance < min_distance)
			{
				min_distance = distance;
				// observations[i].id = predicted[j].id;
				closest_observation = j;

			}
		}
		observations[i].id = predicted[closest_observation].id;
		// observations[i].x = predicted[closest_observation].x;
		// observations[i].y = predicted[closest_observation].y;

	}
}








void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	int number_observed_landmarks = observations.size();
	for(int i = 0; i < num_particles; i++)
	{
    // Select only the map landmarks that are within range of the particles
		std::vector<LandmarkObs> in_range_landmarks;
		for(int j = 0; j < map_landmarks.landmark_list.size(); j++)
		{
			double distance = dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f);
			if(distance <= sensor_range)
			{
				in_range_landmarks.push_back( LandmarkObs{map_landmarks.landmark_list[j].id_i, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f});
			}
		}





		// Transform the observation from car coordinates to map coordinates
		std::vector<LandmarkObs> transformed_observations;
		for(int j = 0; j < observations.size(); j++)
		{
			transformed_observations.push_back(homegenous_transformation(particles[i], observations[j]));
		}

		// Get the landmarks correspondng to closest of observed landmark
		// The recommended function dataAssociation is not very usefull, since even if we have the id to make the correspondence,
		// We will need to search for (almost) the whole data set of in range landmarks AGAIN, for each observations
		std::vector<LandmarkObs> corresponded_landmarks =  corresponding_landmarks(in_range_landmarks, transformed_observations);


		// Calculate the weight the particle by the calculating the multivariable Gaussian distribution
		double probability = 1.0;
		for(int j = 0; j < transformed_observations.size(); j++)
		{

			double temp_prob = multivariable_gaussian_probability(corresponded_landmarks[j].x, corresponded_landmarks[j].y, transformed_observations[j].x, transformed_observations[j].y, std_landmark[0], std_landmark[1]);
			if(temp_prob <= 0.00001) temp_prob = 0.00001;
			probability *= temp_prob;

		}



		particles[i].weight = probability;

	}

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution


  std::random_device rd;
  std::mt19937 gen(rd());
	std::cout <<  weights.size() << '\n';


	// Get the sum of all the weights to normalize
	double weights_sum = 0;
	for(int i = 0; i < num_particles; i++)
	{
			weights_sum += particles[i].weight;
	}


	// Create a particle vector to where I will push the sampled particles
	std::vector<Particle> new_particles;

	// For num_particles times, I will create a random number between 0 and 1.
	// And then associate a particle
	for(int i = 0; i < num_particles; i++)
	{
			double random_weight = (double)rand()/RAND_MAX;

			// Find the 'appropriate' particle
			for(int j = 0; j < num_particles; j++)
			{
				if(particles[j].weight/weights_sum > random_weight)
				{
					new_particles.push_back(particles[j]);
					break;
				}else{
					random_weight -= particles[j].weight;
				}

			}
	}

	// Save the new set of particles
	particles = new_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}


















std::vector<LandmarkObs> ParticleFilter::corresponding_landmarks(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> observations)
{
	// This function is almost identical with the dataAssociation, but the output is much more useful

	int number_predicted_landmarks = predicted.size();
	int number_observed_landmarks = observations.size();
	std::vector<LandmarkObs> corresponded_landmarks;

	// Variables to store the minimum distance found for each observed landmark
	double min_distance, distance;

	for(int i = 0; i < number_observed_landmarks; i++)
	{

		min_distance = std::numeric_limits<double>::infinity();
		int closest_landmark = 0;

		for(int j = 0; j < number_predicted_landmarks; j++)
		{
			distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
			if(distance < min_distance)
			{
				min_distance = distance;
				closest_landmark = j;

			}
		}
		observations[i].id = predicted[closest_landmark].id;
		// observations[i].x = predicted[closest_landmark].x;
		// observations[i].y = predicted[closest_landmark].y;
		corresponded_landmarks.push_back(predicted[closest_landmark]);
	}

	return corresponded_landmarks;

}









/*
 * Perform homegenous transformation from the particle coordinate system to the map coordinate system
 */
LandmarkObs homegenous_transformation(Particle particle, LandmarkObs observed_landmark)
{
	LandmarkObs transformed_landmark;

	transformed_landmark.id = observed_landmark.id;
	transformed_landmark.x = cos(particle.theta)*observed_landmark.x - sin(particle.theta)*observed_landmark.y + particle.x;
	transformed_landmark.y = sin(particle.theta)*observed_landmark.x + cos(particle.theta)*observed_landmark.y + particle.y;

	return transformed_landmark;

}


/*
 * Calculate the multivariable Gaussian probability for a 2 dimesions variable
*/
double multivariable_gaussian_probability(double x1, double y1, double x2, double y2, double sigma_x, double sigma_y)
{
	double probability = exp(-(x1 - x2)*(x1 - x2)/2/(sigma_x*sigma_x) - (y1 - y2)*(y1 - y2)/2/(sigma_y*sigma_y));
	probability = probability/2/M_PI/sigma_x/sigma_y;

	return probability;
}
