#include "visp_odom/visp_odom.hpp"

#if defined(VISP_HAVE_QUALISYS) && (VISP_CXX_STANDARD >= VISP_CXX_STANDARD_11)

VispOdom::VispOdom() : Node("trajectory_publisher") {

	timesync_sub_ = this->create_subscription<px4_msgs::msg::Timesync>("/fmu/timesync/out",
    10,
    [this](const px4_msgs::msg::Timesync::UniquePtr msg) {
        timestamp_.store(msg->timestamp);
    });

	// Publishers
	odom_publisher_ = this->create_publisher<px4_msgs::msg::VehicleVisualOdometry>("/fmu/vehicle_visual_odometry/in", 1);
	
	qualisys_.setVerbose(false);
  	qualisys_.setServerAddress("192.168.30.42");
  	
  	timestamp_sample = 0;

	publisher();
}

void VispOdom::publisher(){

	auto timer_callback = [this]() -> void {

  		if (!qualisys_.connect()) {
    		std::cout << "Qualisys connection error" << std::endl;
    		return;
  		}
  		
  		if(rclcpp::ok()){
  			      
      		if (!qualisys_.getSpecificBodyPose("Acanthis_5", pose_)) {
			std::cout << "Qualisys error. Check the Qualisys Task Manager" << std::endl;
      		}
      		position_ = pose_.getTranslationVector();
      		timestamp_sample++;
      		
      		odometry_.timestamp = timestamp_.load();
      		odometry_.timestamp_sample = timestamp_.load();
      		odometry_.local_frame = 0; //NED
      		odometry_.x = position_[1];
      		odometry_.y = position_[0];
      		odometry_.z = -position_[2];
      		
      		//std::cout << "P: " << position_.t() << std::endl;
      		
      		odom_publisher_->publish(odometry_);
  		}
  
	};
	timer_ = this->create_wall_timer(20ms, timer_callback); // 50 Hz
}


int main(int argc, char* argv[]) {
	std::cout << "Starting visp_odom node..." << std::endl;
	setvbuf(stdout, NULL, _IONBF, BUFSIZ);
	rclcpp::init(argc, argv);

	rclcpp::spin(std::make_shared<VispOdom>());

	rclcpp::shutdown();
	return 0;
}
#else
int main()
{
  std::cout << "ViSP doesn't support Qualisys" << std::endl;
  return EXIT_SUCCESS;
}

#endif

