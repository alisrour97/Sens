#include "trajectory_publisher/trajectory_publisher.hpp"

inline Eigen::Matrix3f QuatToMat(Eigen::Vector4f Quat){
    Eigen::Matrix3f Rot;
    float s = Quat[0];
    float x = Quat[1];
    float y = Quat[2];
    float z = Quat[3];
    Rot << 1-2*(y*y+z*z),2*(x*y-s*z),2*(x*z+s*y),
    2*(x*y+s*z),1-2*(x*x+z*z),2*(y*z-s*x),
    2*(x*z-s*y),2*(y*z+s*x),1-2*(x*x+y*y);
    return Rot;
}


TrajectoryPublisher::TrajectoryPublisher() : Node("trajectory_publisher") {

	declare_parameter("file_name","drone_data.csv");

	file_name_ = get_parameter("file_name").as_string();

	std::cout << "Loaded " << file_name_ << std::endl;

	rmw_qos_profile_t qos_profile = rmw_qos_profile_sensor_data;
	auto qos = rclcpp::QoS(rclcpp::QoSInitialization(qos_profile.history, 0), qos_profile);

	timesync_sub_ = this->create_subscription<px4_msgs::msg::Timesync>("/fmu/timesync/out",10,
		[this](const px4_msgs::msg::Timesync::UniquePtr msg) {
			timestamp_.store(msg->timestamp);
		}
	);

	// Publishers
	offboard_control_mode_publisher_ = this->create_publisher<OffboardControlMode>("/fmu/offboard_control_mode/in", 10);
	trajectory_setpoint_publisher_ = this->create_publisher<TrajectorySetpoint>("/fmu/trajectory_setpoint/in", 10);
	vehicle_command_publisher_ = this->create_publisher<VehicleCommand>("/fmu/vehicle_command/in", 10);

	// Subscribers
    odom_sub_ = this->create_subscription<px4_msgs::msg::VehicleOdometry>("/fmu/vehicle_odometry/out",qos,
                    std::bind(&TrajectoryPublisher::odomFeedback, this, std::placeholders::_1));


	traj_cnt_ = 0;

	load_trajectory();

	takeoff_pos_(2) = des_pos_(0, 2);

	offboard_setpoint_counter_ = 0;

	landed_ = false;
	takeoff_ = false;
	armed_ = false;



	publisher();
}


/**
 * @brief Send a command to Arm the vehicle
 */
void TrajectoryPublisher::arm(){
	publish_vehicle_command(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0);

	RCLCPP_INFO(this->get_logger(), "Arm command send");
}

/**
 * @brief Send a command to Disarm the vehicle
 */
void TrajectoryPublisher::disarm(){
	publish_vehicle_command(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0);

	RCLCPP_INFO(this->get_logger(), "Disarm command send");
}

/**
 * @brief Publish the offboard control mode.
 *        We will send position, velocity and acceleration
 */
void TrajectoryPublisher::publish_offboard_control_mode(){
	OffboardControlMode msg{};
	msg.timestamp = timestamp_.load();
	msg.position = true;
	msg.velocity = true;
	msg.acceleration = true;
	msg.attitude = false;
	msg.body_rate = false;

	offboard_control_mode_publisher_->publish(msg);
}

void TrajectoryPublisher::takeoff(){
	TrajectorySetpoint msg{};
	msg.x = takeoff_pos_(0);
	msg.y = takeoff_pos_(1);
	msg.z = takeoff_pos_(2);
	msg.yaw = takeoff_yaw_; // [-PI:PI]
	msg.timestamp = timestamp_.load();

	trajectory_setpoint_publisher_->publish(msg);

}

void TrajectoryPublisher::land(){
	TrajectorySetpoint msg{};
	msg.x = land_pos_(0);
	msg.y = land_pos_(1);
	msg.z = land_pos_(2);
	msg.yaw = yaw_; // [-PI:PI]
	msg.timestamp = timestamp_.load();

	trajectory_setpoint_publisher_->publish(msg);

}

/**
 * @brief Publish vehicle commands
 * @param command   Command code (matches VehicleCommand and MAVLink MAV_CMD codes)
 * @param param1    Command parameter 1
 * @param param2    Command parameter 2
 */
void TrajectoryPublisher::publish_vehicle_command(uint16_t command, float param1,
					      float param2){
	VehicleCommand msg{};
	msg.timestamp = timestamp_.load();
	msg.param1 = param1;
	msg.param2 = param2;
	msg.command = command;
	msg.target_system = 1;
	msg.target_component = 1;
	msg.source_system = 1;
	msg.source_component = 1;
	msg.from_external = true;

	vehicle_command_publisher_->publish(msg);
}

void TrajectoryPublisher::load_trajectory(){

	
	std::vector<std::vector<std::string>> content;
	std::vector<std::string> row;
	std::string line, word;
	std::string file_name = "src/trajectory_publisher/csv_file/"+file_name_;
	std::fstream traj_file (file_name, std::ios::in);
	if(traj_file.is_open())
	{
		while(std::getline(traj_file, line))
		{
			row.clear();
 
			std::stringstream str(line);
 
			while(std::getline(str, word, ','))
				row.push_back(word);

			content.push_back(row);
		}

		trajectory_setpoints_ = content.size();

		// Avoid the first row of titles
		for(int i=1; i < trajectory_setpoints_ ;i++)
		{
			traj_time_.conservativeResize(traj_time_.size()+1);
			des_pos_.conservativeResize(des_pos_.rows()+1, des_pos_.cols());
			des_vel_.conservativeResize(des_vel_.rows()+1, des_vel_.cols());
			des_acc_.conservativeResize(des_acc_.rows()+1, des_acc_.cols());
			des_yaw_.conservativeResize(des_yaw_.size()+1);
			des_yaw_rate_.conservativeResize(des_yaw_rate_.size()+1);

			// The csv trajectory is in ROTATED FLU (Qualisys) and here is transformed in NED
			traj_time_(i-1) = (float)std::atof(content[i][0].c_str());
			des_pos_.row(i-1) << (float)std::atof(content[i][2].c_str()), (float)std::atof(content[i][1].c_str()), -(float)std::atof(content[i][3].c_str());
			des_yaw_(i-1) = -(float)std::atof(content[i][4].c_str());
			des_vel_.row(i-1) << (float)std::atof(content[i][6].c_str()), (float)std::atof(content[i][5].c_str()), -(float)std::atof(content[i][7].c_str());
			des_yaw_rate_(i-1) = -(float)std::atof(content[i][8].c_str());
			des_acc_.row(i-1) << (float)std::atof(content[i][10].c_str()), (float)std::atof(content[i][9].c_str()), -(float)std::atof(content[i][11].c_str());
			
			// std::cout << traj_time(i-1) << ") " << des_pos.row(i-1) << " " << des_yaw(i-1) << " " << des_vel.row(i-1) << std::endl;
		}
	}
	else{
		std::cout<<"Could not open the file\n";
		exit(0);
	}

}

void TrajectoryPublisher::publisher(){

	auto timer_callback = [this]() -> void {

		if (offboard_setpoint_counter_ == 10) {
			// Change to Offboard mode after 10 setpoints
			this->publish_vehicle_command(VehicleCommand::VEHICLE_CMD_DO_SET_MODE, 1, 6);

			// Arm the vehicle
			this->arm();
			takeoff_pos_ << pos_(0), pos_(1), des_pos_(0,2);
			takeoff_yaw_ = yaw_;
			armed_ = true;
		}
		else if(armed_){
			
			if(fabs(pos_(2)) < fabs(takeoff_pos_(2)) && !takeoff_){
				//this->publish_vehicle_command(VehicleCommand::VEHICLE_CMD_DO_SET_MODE, 1, 6);
				publish_offboard_control_mode();
				std::cout << "Taking OFF" << std::endl;
				this->takeoff();
			}
			else if(traj_cnt_ < trajectory_setpoints_-1){
				takeoff_ = true;

				traj_sp_.x = des_pos_(traj_cnt_,0);
				traj_sp_.y = des_pos_(traj_cnt_,1);
				traj_sp_.z = des_pos_(traj_cnt_,2);
				traj_sp_.vx = des_vel_(traj_cnt_,0);
				traj_sp_.vy = des_vel_(traj_cnt_,1);
				traj_sp_.vz = des_vel_(traj_cnt_,2);
				traj_sp_.acceleration = {des_acc_(traj_cnt_,0), des_acc_(traj_cnt_,1), des_acc_(traj_cnt_,2)};
				traj_sp_.yaw = des_yaw_(traj_cnt_);
				traj_sp_.timestamp = timestamp_.load();

				publish_offboard_control_mode();
				trajectory_setpoint_publisher_->publish(traj_sp_);

				traj_cnt_++;

				if(traj_cnt_ == trajectory_setpoints_-1) {
					land_pos_ << pos_(0), pos_(1), 0.5;
				}

			}
			else{
				if(!landed_){
					publish_offboard_control_mode();
					std::cout << "LANDING" << std::endl;
					this->land();
					publish_vehicle_command(VehicleCommand::VEHICLE_CMD_NAV_LAND);

					RCLCPP_INFO(this->get_logger(), "Land command send");
					landed_ = true;
				}
			}
		}
		if(offboard_setpoint_counter_ < 11) {
			traj_sp_.x = pos_(0);
			traj_sp_.y = pos_(1);
			traj_sp_.z = pos_(2);
			traj_sp_.vx = 0.0;
			traj_sp_.vy = 0.0;
			traj_sp_.vz = 0.0;
			traj_sp_.acceleration = {0.0, 0.0, 0.0};
			traj_sp_.yaw = yaw_;
			traj_sp_.timestamp = timestamp_.load();

			publish_offboard_control_mode();
			trajectory_setpoint_publisher_->publish(traj_sp_);
			offboard_setpoint_counter_++;
		}
	};
	timer_ = this->create_wall_timer(20ms, timer_callback); // 50 Hz
}

void TrajectoryPublisher::odomFeedback(px4_msgs::msg::VehicleOdometry::UniquePtr msg){
    
    pos_ << msg->x, msg->y, msg->z;
    vel_ << msg->vx, msg->vy, msg->vz;
    
	//std::cout << "pos: " << pos_.transpose() << std::endl;

    R_att_ =  QuatToMat( Eigen::Vector4f( msg->q[0], msg->q[1], msg->q[2], msg->q[3] ));
    w_att_ << msg->rollspeed, msg->pitchspeed, msg->yawspeed;

	yaw_ = atan2(R_att_(1,0),R_att_(0,0));

    // pos_ = R_att_ * pos_;
    // vel_ = R_att_ * vel_;
   
}

int main(int argc, char* argv[]) {
	std::cout << "Starting trajectory_publisher node..." << std::endl;
	setvbuf(stdout, NULL, _IONBF, BUFSIZ);
	rclcpp::init(argc, argv);

	rclcpp::spin(std::make_shared<TrajectoryPublisher>());

	rclcpp::shutdown();
	return 0;
}
