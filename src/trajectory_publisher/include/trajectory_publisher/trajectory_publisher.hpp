#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/vehicle_control_mode.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <px4_msgs/msg/timesync.hpp>
#include <rclcpp/rclcpp.hpp>
#include <Eigen/Eigen>
#include <stdint.h>

#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

using namespace std::chrono;
using namespace std::chrono_literals;
using namespace px4_msgs::msg;

class TrajectoryPublisher : public rclcpp::Node {
public:
	
    TrajectoryPublisher();
	void arm();
	void disarm();
	void takeoff();
	void land();

private:

    void publish_offboard_control_mode();
	void publish_vehicle_command(uint16_t command, float param1 = 0.0, float param2 = 0.0);
	void odomFeedback(const px4_msgs::msg::VehicleOdometry::UniquePtr msg);
    void load_trajectory();
    void publisher();

	std::string file_name_;

	rclcpp::TimerBase::SharedPtr timer_;

	rclcpp::Publisher<OffboardControlMode>::SharedPtr offboard_control_mode_publisher_;
	rclcpp::Publisher<TrajectorySetpoint>::SharedPtr trajectory_setpoint_publisher_;
	rclcpp::Publisher<VehicleCommand>::SharedPtr vehicle_command_publisher_;
	rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr odom_sub_;
	rclcpp::Subscription<px4_msgs::msg::Timesync>::SharedPtr timesync_sub_;


	std::atomic<uint64_t> timestamp_;   //!< common synced timestamped

	float takeoff_yaw_;


	Eigen::Vector3f takeoff_pos_; // drone takeoff linear position
	Eigen::Vector3f land_pos_; // drone landing linear position
	Eigen::Vector3f pos_; // drone linear position
    Eigen::Vector3f vel_; // drone linear velocity
    Eigen::Matrix3f R_att_; // drone attitude rotation matrix
    Eigen::Vector3f w_att_; // drone angular rate
	float yaw_;

	bool landed_;
	bool takeoff_;
	bool armed_;

    int traj_cnt_, offboard_setpoint_counter_;
    TrajectorySetpoint traj_sp_{};

    int trajectory_setpoints_;
    Eigen::VectorXf traj_time_, des_yaw_, des_yaw_rate_;
	Eigen::Matrix<float, Eigen::Dynamic, 3> des_pos_, des_vel_, des_acc_;

	
};
