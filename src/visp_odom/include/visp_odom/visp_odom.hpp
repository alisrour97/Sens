#include <px4_msgs/msg/vehicle_visual_odometry.hpp>
#include <px4_msgs/msg/timesync.hpp>
#include <rclcpp/rclcpp.hpp>
#include <stdint.h>
#include <visp3/sensor/vpMocapQualisys.h>
#if defined(VISP_HAVE_QUALISYS) && (VISP_CXX_STANDARD >= VISP_CXX_STANDARD_11)
 
#include <visp3/core/vpTime.h>

#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

using namespace std::chrono;
using namespace std::chrono_literals;
using namespace px4_msgs::msg;

class VispOdom : public rclcpp::Node {
public:
	
    VispOdom();
	
private:

	void publisher();

	rclcpp::TimerBase::SharedPtr timer_;

	rclcpp::Publisher<px4_msgs::msg::VehicleVisualOdometry>::SharedPtr odom_publisher_;
	rclcpp::Subscription<px4_msgs::msg::Timesync>::SharedPtr timesync_sub_;

	std::atomic<uint64_t> timestamp_;   //!< common synced timestamped

	std::atomic<uint64_t> timestamp_sample;
	
	VehicleVisualOdometry odometry_{};
	
	vpMocapQualisys qualisys_;
	vpHomogeneousMatrix pose_;
	vpTranslationVector position_;
	vpRotationMatrix rot_mat_, rot_mat_visp2NED_;
	vpQuaternionVector quat_;
	vpRxyzVector eul_;
	double yaw_;

};

#endif

