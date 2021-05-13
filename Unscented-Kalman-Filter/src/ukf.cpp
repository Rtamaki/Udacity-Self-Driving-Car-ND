#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = false;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = false;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 3;

  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  n_x_ = x_.size();
  n_aug_ = n_x_ + 2;
  Xsig_pred_ = MatrixXd(n_x_, n_aug_ * 2 + 1);
  weights_ = VectorXd(2*n_aug_ + 1);

  double weight_0 = lambda_/(lambda_ + n_aug_);
  weights_(0) = weight_0;
  for (int i = 1; i < 2*n_aug_ + 1; i++) {
      double weight = 0.5/(n_aug_ + lambda_);
      weights_(i) = weight;
 }




}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
    TODO:
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
    // first measurement
    cout << "EKF: " << endl;
    x_ = VectorXd(5);
    x_ << 1, 1, 1, 0, 0;
    P_(0, 0) = 1;
    P_(1, 1) = 1;
    P_(2, 2) = 1;
    P_(3, 3) = 1;
    P_(4, 4) = 1;

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      float range = meas_package.raw_measurements_[0];
      float phi = meas_package.raw_measurements_[1];
      float range_dot = meas_package.raw_measurements_[2];
      x_ << range * cos(phi), range * sin(phi), range_dot * cos(phi), phi, 0;

    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
    }

    time_us_ = meas_package.timestamp_;
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }


  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  float dt = (meas_package.timestamp_ - time_us_) / 1000000.0;	//dt - expressed in seconds
  time_us_ = meas_package.timestamp_;
  Prediction(dt);



  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
   TODO:
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    UpdateRadar(meas_package);
    cout << "RADAR" << endl;


  } else {
    // Laser updates
    UpdateLidar(meas_package);
    cout << "LIDAR" << endl;

  }
  // print the output
  cout << "x_ = " << x_ << endl;
  cout << "P_ = " << P_ << endl;


}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

  SigmaPointPrediction(delta_t);
  PredictMeanAndCovariance();

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */


    //set measurement dimension, radar can measure r, phi, and r_dot
    int n_z = 2;

    //create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

    //mean predicted measurement
    VectorXd z_pred = VectorXd(n_z);

    //measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z, n_z);

    //transform sigma points into measurement space
    for(int i = 0; i < 2 * n_aug_ + 1; i++)
    {
        double px = Xsig_pred_(0,i);
        double py = Xsig_pred_(1,i);
        double v = Xsig_pred_(2,i);
        double phi = Xsig_pred_(3,i);
        double phi_dot = Xsig_pred_(4,i);

        Zsig.col(i) << px,
                       py;
    }

    //calculate mean predicted measurement
    z_pred = Zsig*weights_;

    //calculate innovation covariance matrix S
    S.fill(0.0);
    for(int i = 0; i < 2 * n_aug_ + 1; i++)
    {
        VectorXd diff = VectorXd(n_z);
        diff = Zsig.col(i) - z_pred;
        S = S + weights_(i)*diff*diff.transpose();
    }

    MatrixXd R = MatrixXd(n_z, n_z);
    R(0,0) = std_laspx_*std_laspx_;
    R(1,1) = std_laspy_*std_laspy_;

    S = S + R;

    //create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, n_z);

    //calculate cross correlation matrix
    Tc.fill(0.0);
    for(int i = 0; i < 2*n_aug_ + 1; i++)
    {
          Tc = Tc + weights_(i)*(Xsig_pred_.col(i) - x_) * (Zsig.col(i) - z_pred).transpose();
    }

    //calculate Kalman gain K;
    MatrixXd K = MatrixXd(n_x_, n_z);
    K = Tc*S.inverse();

    //update state mean and covariance matrix
    x_ = x_ + K * (meas_package.raw_measurements_ - z_pred);
    P_ = P_ - K * S * K.transpose();

    // Guarantee that the estimated Phi angle is within
    while(x_(3) < -3.1415 || x_(3) > 3.1415)
    {
      if(x_(3) < -3.1415){
        x_(3) = x_(3) + 2*3.1415;
      } else
      {
        if(x_(3) > 3.1415){
          x_(3) = x_(3) - 2 * 3.1415;
        }
      }
    }



}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */


  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);

  //transform sigma points into measurement space
  for(int i = 0; i < 2 * n_aug_ + 1; i++)
  {
      double px = Xsig_pred_(0,i);
      double py = Xsig_pred_(1,i);
      double v = Xsig_pred_(2,i);
      double phi = Xsig_pred_(3,i);
      double phi_dot = Xsig_pred_(4,i);

      Zsig.col(i) << sqrt(px*px + py*py),
                     atan2(py, px),
                     (px*cos(phi)*v + py*sin(phi)*v)/sqrt(px*px + py*py);
  }

  //calculate mean predicted measurement
  z_pred = Zsig*weights_;

  //calculate innovation covariance matrix S
  S.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; i++)
  {
      VectorXd diff = VectorXd(n_z);
      diff = Zsig.col(i) - z_pred;
      S = S + weights_(i)*diff*diff.transpose();
  }

  MatrixXd R = MatrixXd(n_z,n_z);
  R(0,0) = std_radr_*std_radr_;
  R(1,1) = std_radphi_*std_radphi_;
  R(2,2) = std_radrd_*std_radrd_;

  S = S + R;

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for(int i = 0; i < 2*n_aug_ + 1; i++)
  {
        Tc = Tc + weights_(i)*(Xsig_pred_.col(i) - x_) * (Zsig.col(i) - z_pred).transpose();
  }

  //calculate Kalman gain K;
  MatrixXd K = MatrixXd(n_x_, n_z);
  K = Tc*S.inverse();

  //update state mean and covariance matrix
  x_ = x_ + K * (meas_package.raw_measurements_ - z_pred);
  P_ = P_ - K * S * K.transpose();

  while(x_(3) < -3.1415 || x_(3) > 3.1415)
  {
    if(x_(3) < -3.1415){
      x_(3) = x_(3) + 2*3.1415;
    } else
    {
      if(x_(3) > 3.1415){
        x_(3) = x_(3) - 2 * 3.1415;
      }
    }
  }

}



/**
  * Functions necessary for Unscented Kalman Filter Predition
  */

void UKF::GenerateSigmaPoints(MatrixXd* Xsig_out) {

  //create sigma point matrix
  MatrixXd Xsig = MatrixXd(n_x_, 2 * n_x_ + 1);

  //calculate square root of P
  MatrixXd A = (P_.llt()).matrixL();


  // calculate sigma points
  Xsig << x_, (sqrt(n_x_ + lambda_)*A).colwise() + x_ , (- (sqrt(n_x_ + lambda_)*A)).colwise() + x_;

  *Xsig_out = Xsig;

}

void UKF::AugmentedSigmaPoints(MatrixXd* Xsig_out) {

  // Create augmented state vector
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug << x_,
           0,
           0;

  // Create noise covariance matrix
  MatrixXd Q = MatrixXd(n_aug_ - n_x_, n_aug_ - n_x_);
  Q << std_a_*std_a_, 0,
       0,             std_yawdd_*std_yawdd_;

  // Create augemented covariance matrix
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  //auxiliar matrix of zeros
  MatrixXd aux_zeros = MatrixXd::Zero(n_x_, n_aug_ - n_x_);
  P_aug << P_ ,               aux_zeros,
           aux_zeros.transpose(), Q;

   //create square root matrix
   MatrixXd A = P_aug.llt().matrixL();

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  Xsig_aug << x_aug, (sqrt(n_aug_ + lambda_)*A).colwise() + x_aug , (- (sqrt(n_aug_ + lambda_)*A)).colwise() + x_aug ;

  //write result
  *Xsig_out = Xsig_aug;

}

void UKF::SigmaPointPrediction(double delta_t){


  MatrixXd Xsig_aug;
  AugmentedSigmaPoints (&Xsig_aug);

  //predict sigma points
  Xsig_pred_ << Xsig_aug.block(0, 0, n_x_, 2 * n_aug_ + 1);


  for(int i = 0; i < 2 * n_aug_ + 1; i++)
  {
        double px = Xsig_aug(0,i);
        double py = Xsig_aug(1,i);
        double v = Xsig_aug(2,i);
        double phi = Xsig_aug(3,i);
        double phi_dot = Xsig_aug(4,i);
        double mi_a = Xsig_aug(5,i);
        double mi_phi = Xsig_aug(6,i);

        double px_p, py_p, v_p, phi_p, phi_dot_p;
        // std::cout << sin(1) << std::endl;

        // Calculate the steps
        if(fabs(phi_dot) > 0.001)
        {
            px_p = px + v/phi_dot*(sin(phi + phi_dot*delta_t) - sin(phi));
            py_p = py + v/phi_dot*(-cos(phi + phi_dot*delta_t) + cos(phi));
            v_p = v + 0;
            phi_p = phi + phi_dot*delta_t;
            phi_dot_p = phi_dot + 0;

        } else
        {
            px_p = px + v*cos(phi)*delta_t;
            py_p = py + v*sin(phi)*delta_t;
            v_p = v + 0,
            phi_p = phi + phi_dot*delta_t,
            phi_dot_p = phi_dot + 0;
        }

        px_p += 0.5 * delta_t*delta_t*cos(phi)*mi_a;
        py_p += 0.5 * delta_t*delta_t*sin(phi)*mi_a;
        v_p += delta_t*mi_a;
        phi_p += 0.5*delta_t*delta_t*mi_phi;
        phi_dot_p += delta_t*mi_phi;


        Xsig_pred_(0,i) = px_p;
        Xsig_pred_(1,i) = py_p;
        Xsig_pred_(2,i) = v_p;
        Xsig_pred_(3,i) = phi_p;
        Xsig_pred_(4,i) = phi_dot_p;
  }
}

void UKF::PredictMeanAndCovariance() {


  //  Predict state mean
  x_.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; i++)
  {
      double w;
      if(i == 0){
          w = lambda_/(lambda_ + n_aug_);
      }else
      {
          w = 0.5/(lambda_ + n_aug_);
      }

      x_ = x_ + Xsig_pred_.col(i)*w;
  }

  // Normalize the angle phi
  while(x_(3) < -3.1415 || x_(3) > 3.1415)
  {
    if(x_(3) < -3.1415){
      x_(3) = x_(3) + 2*3.1415;
    } else
    {
      if(x_(3) > 3.1415){
        x_(3) = x_(3) - 2 * 3.1415;
      }
    }
  }

  //  Predict the covariance matrix
  P_.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; i++)
  {
      double w;
      if(i == 0){
          w = lambda_/(lambda_ + n_aug_);
      }else
      {
          w = 0.5/(lambda_ + n_aug_);
      }

      VectorXd x_diff = VectorXd(n_x_);
      x_diff = Xsig_pred_.col(i) - x_;
      P_ = P_ + w*(x_diff * x_diff.transpose());

  }
}







void UKF::PredictRadarMeasurement(VectorXd* z_out, MatrixXd* S_out) {

  int n_z = 3;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);

  //transform sigma points into measurement space
  for(int i = 0; i < 2 * n_aug_ + 1; i++)
  {
      double px = Xsig_pred_(0,i);
      double py = Xsig_pred_(1,i);
      double v = Xsig_pred_(2,i);
      double phi = Xsig_pred_(3,i);
      double phi_dot = Xsig_pred_(4,i);

      Zsig.col(i) << sqrt(px*px + py*py),
                     atan2(py, px),
                     (px*cos(phi)*v + py*sin(phi)*v)/sqrt(px*px + py*py);
  }

  //calculate mean predicted measurement
  z_pred = Zsig*weights_;

  //calculate innovation covariance matrix S
  S.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; i++)
  {
      VectorXd diff = VectorXd(n_z);
      diff = Zsig.col(i) - z_pred;
      S = S + weights_(i)*diff*diff.transpose();
  }

  MatrixXd R = MatrixXd(n_z,n_z);
  R(0,0) = std_radr_*std_radr_;
  R(1,1) = std_radphi_*std_radphi_;
  R(2,2) = std_radrd_*std_radrd_;

  S = S + R;

  *z_out = z_pred;
  *S_out = S;

}

void UKF::PredictLidarMeasurement(VectorXd* z_out, MatrixXd* S_out) {

  int n_z = 2;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);

  //transform sigma points into measurement space
  for(int i = 0; i < 2 * n_aug_ + 1; i++)
  {
      double px = Xsig_pred_(0,i);
      double py = Xsig_pred_(1,i);

      Zsig.col(i) << px,
                     py;
  }

  //calculate mean predicted measurement
  z_pred = Zsig*weights_;

  //calculate innovation covariance matrix S
  S.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; i++)
  {
      VectorXd diff = VectorXd(n_z);
      diff = Zsig.col(i) - z_pred;
      S = S + weights_(i)*diff*diff.transpose();
  }

  MatrixXd R = MatrixXd(n_z,n_z);
  R(0,0) = std_laspx_*std_laspx_;
  R(1,1) = std_laspy_*std_laspy_;

  S = S + R;

  *z_out = z_pred;
  *S_out = S;

}
