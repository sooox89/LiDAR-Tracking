#include "utils.hpp"

class EgoLocalization
{
public:
    double roll;
    double pitch;
    double yaw;  // rad

    double vx;   // 동차계 x방향 속도
    double vy;   // 동차계 y방향 속도
    double v;    // 속도 크기 (m/s)

    EgoLocalization()
        : roll(0.0), pitch(0.0), yaw(0.0),
          vx(0.0), vy(0.0), v(0.0)
    {
    }

    void update(const novatel_oem7_msgs::INSPVA::ConstPtr& msg)
    {
        // 자세 정보
        roll = msg->roll;
        pitch = msg->pitch;
        double yaw_deg = fmod((90.0 - msg->azimuth), 360.0); // 북쪽 기준
        yaw = yaw_deg * M_PI / 180.0;

        // 속도 계산
        calculateVelocityAndHeading(msg);
    }

private:
    void calculateVelocityAndHeading(const novatel_oem7_msgs::INSPVA::ConstPtr& msg)
    {
        double azimuth_rad = msg->azimuth * M_PI / 180.0;
        double cos_azimuth = std::cos(azimuth_rad);
        double sin_azimuth = std::sin(azimuth_rad);

        vx = msg->north_velocity * cos_azimuth + msg->east_velocity * sin_azimuth;
        vy = -msg->north_velocity * sin_azimuth + msg->east_velocity * cos_azimuth;

        v = std::sqrt(vx * vx + vy * vy);
    }
};

class Tracking {
public:
    Tracking() {};

    Tracking(ros::NodeHandle& nh) : nh_(nh) {
        // ROS Parameters
        nh_.getParam("Public/lidar_frame", lidar_frame);
        nh_.getParam("Public/target_frame", target_frame);
        nh_.getParam("Public/world_frame", world_frame);
        nh_.getParam("Tracking/integration/mode", mode);
        nh_.getParam("Tracking/integration/thresh_iou", thresh_iou);
        nh_.getParam("Tracking/crop_hd_map/radius", radius);
        nh_.getParam("Tracking/track/invisibleCnt", invisibleCnt);
        nh_.getParam("Tracking/track/deque/number_velocity", number_velocity_deque);
        nh_.getParam("Tracking/track/deque/number_orientation", number_orientation_deque);
        nh_.getParam("Tracking/track/deque/thresh_velocity", thresh_velocity);
        nh_.getParam("Tracking/track/deque/thresh_orientation", thresh_orientation);
        nh_.getParam("Tracking/postProcessing/thresh_x_distance", thresh_x_distance);
        nh_.getParam("Tracking/postProcessing/thresh_y_distance", thresh_y_distance);
        nh_.getParam("Tracking/postProcessing/thresh_predictSec", thresh_predictSec);

        // mission
        nh_.getParam("Mission/tracking/cluster_distance", cluster_distance);
        nh_.getParam("Mission/tracking/ground_removal_distance", ground_removal_distance);
        nh_.getParam("Mission/tracking/cluster_size", cluster_size);
        nh_.getParam("Mission/tracking/cluster_ratio", cluster_ratio);
        nh_.getParam("Mission/tracking/deep_distance", deep_distance);
        nh_.getParam("Mission/tracking/deep_score", deep_score);
        
        global_path = map_reader(map.c_str());
        
        // integration
        last_timestamp_cluster = -1;
        last_timestamp_deep = -1;
        
        tracker.setParams(invisibleCnt, number_velocity_deque, number_orientation_deque, thresh_velocity, thresh_orientation);

        waypoints_cache.setCacheSize(600);

        clearLogFile(integration_time_log_path);
        clearLogFile(crophdmap_time_log_path);
        clearLogFile(tracking_time_log_path);
        clearLogFile(transform_time_log_path);
        clearLogFile(correction_time_log_path);
    }

    void updateWaypoints(const sensor_msgs::PointCloud2::ConstPtr &cloud_msg);
    void integrationBbox(jsk_recognition_msgs::BoundingBoxArray &cluster_bbox_array, 
                         jsk_recognition_msgs::BoundingBoxArray &deep_bbox_array,
                         jsk_recognition_msgs::BoundingBoxArray &output_bbox_array, 
                         std::vector<bool> &deep_check_array,
                         double &time_taken);
    
    void cropHDMapBbox(const jsk_recognition_msgs::BoundingBoxArray &input_bbox_array, 
                        jsk_recognition_msgs::BoundingBoxArray &output_bbox_array, 
                        const ros::Time &input_stamp, double& time_taken);
    
    void tracking(const jsk_recognition_msgs::BoundingBoxArray &bbox_array, jsk_recognition_msgs::BoundingBoxArray &track_bbox_array, 
                    visualization_msgs::MarkerArray &track_text_array, std::vector<bool> &deep_check_array, const double &egoVehicle_yaw, const ros::Time &input_stamp, double &time_taken);

    void transformBbox(const jsk_recognition_msgs::BoundingBoxArray &input_bbox_array, tf2_ros::Buffer &tf_buffer, 
                        jsk_recognition_msgs::BoundingBoxArray &output_bbox_array, double &time_taken);

    void correctionBboxRelativeSpeed(const jsk_recognition_msgs::BoundingBoxArray &input_bbox_array, const ros::Time &input_stamp, 
                        const ros::Time &cur_stamp, jsk_recognition_msgs::BoundingBoxArray &output_bbox_array, double& time_taken);

    void correctionBboxTF(const jsk_recognition_msgs::BoundingBoxArray &input_bbox_array, const ros::Time &input_stamp, 
                        const ros::Time &cur_stamp, tf2_ros::Buffer &tf_buffer, 
                        jsk_recognition_msgs::BoundingBoxArray &output_bbox_array, double &time_taken);
    
    void postProcessing(const jsk_recognition_msgs::BoundingBoxArray &corrected_bbox_array, lidar_tracking::AdjacentVehicle &msg_PostProcessing, const boost::shared_ptr<EgoLocalization>& st_EgoInfo, double& time_taken);

    void averageTime();

private:
    ros::NodeHandle nh_;
    std::string map;
    std::string lidar_frame;
    std::string target_frame;
    std::string world_frame;
    std::vector<std::pair<float, float>> global_path; // Initialization is Public
    int mode;
    float thresh_iou;    // IOU threshold for bounding box integration
    double radius; // Minimum distance threshold for HD map cropping
    
    int invisibleCnt;
    int number_velocity_deque;
    int number_orientation_deque;
    int thresh_x_distance;
    int thresh_y_distance;
    float thresh_velocity;
    float thresh_orientation;
    float thresh_predictSec;

    // mission
    int cluster_distance;
    int ground_removal_distance;
    float cluster_size;
    float cluster_ratio;
    int deep_distance;
    float deep_score;
    
    double last_timestamp_cluster;
    double last_timestamp_deep;

    Track tracker;

    message_filters::Cache<sensor_msgs::PointCloud2> waypoints_cache;
    
    // average time check
    std::string package_path = ros::package::getPath("lidar_tracking") + "/time_log/tracking/";
    std::string integration_time_log_path = package_path + "integration.txt";
    std::string crophdmap_time_log_path = package_path + "crophdmap.txt";
    std::string tracking_time_log_path = package_path + "tracking.txt";
    std::string transform_time_log_path = package_path + "transform.txt";
    std::string correction_time_log_path = package_path + "correction.txt";
};

void Tracking::updateWaypoints(const sensor_msgs::PointCloud2::ConstPtr &cloud_msg) 
{
    waypoints_cache.add(cloud_msg);
}

void Tracking::integrationBbox(jsk_recognition_msgs::BoundingBoxArray &cluster_bbox_array, 
                               jsk_recognition_msgs::BoundingBoxArray &deep_bbox_array,
                               jsk_recognition_msgs::BoundingBoxArray &output_bbox_array, 
                               std::vector<bool> &deep_check_array,
                               double& time_taken) 
{
    auto start = std::chrono::steady_clock::now();

    output_bbox_array.boxes.clear();
    deep_check_array.clear();
    
    // mode
    // integration : clustering + deep learning
    if (mode == 0) {
        for (const auto &cluster_bbox : cluster_bbox_array.boxes) {
            
            bool keep_cluster_bbox = true;
            for (const auto &deep_bbox : deep_bbox_array.boxes) {
                double overlap = getBBoxOverlap(cluster_bbox, deep_bbox);   // IoU 계산
                // IoU가 thresh_iou보다 큰 경우 Deep Learning 결과를 신뢰
                if (overlap > thresh_iou) {
                    keep_cluster_bbox = false;
                    break;
                }
            }
            // Clustering 결과는 유지
            if (keep_cluster_bbox) {
                output_bbox_array.boxes.push_back(cluster_bbox);
                deep_check_array.push_back(false);
            }
        }
        // Deep Learning 결과는 일단 전부 살림
        output_bbox_array.boxes.insert(output_bbox_array.boxes.end(), deep_bbox_array.boxes.begin(), deep_bbox_array.boxes.end());  
        // Deep Learning 검출 결과의 경우 True로 설정
        size_t num_new_boxes = deep_bbox_array.boxes.size();
        for (int i = 0; i < num_new_boxes; i++) {
            deep_check_array.push_back(true);
}
    } 
    else if (mode == 1) { output_bbox_array = cluster_bbox_array; } 
    else if (mode == 2) { output_bbox_array = deep_bbox_array; }

    last_timestamp_cluster = cluster_bbox_array.header.stamp.toSec();
    last_timestamp_deep = deep_bbox_array.header.stamp.toSec();

    // Integration 과정에서 소요되는 시간
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    time_taken = elapsed_seconds.count();   
    // saveTimeToFile(integration_time_log_path, time_taken);
}

void Tracking::cropHDMapBbox(const jsk_recognition_msgs::BoundingBoxArray &input_bbox_array, 
                             jsk_recognition_msgs::BoundingBoxArray &output_bbox_array, 
                             const ros::Time &input_stamp, double& time_taken) 
{
    auto start = std::chrono::steady_clock::now();

    output_bbox_array.boxes.clear();

    auto closest_waypoint = waypoints_cache.getElemAfterTime(input_stamp);
    if (!closest_waypoint) {
        closest_waypoint = waypoints_cache.getElemBeforeTime(input_stamp);
    }

    if (!closest_waypoint) {
        ROS_WARN("No waypoints found in cache for the specified timestamp.");
        output_bbox_array = input_bbox_array;
        return;
    }

    pcl::PointCloud<pcl::PointXY>::Ptr cloud(new pcl::PointCloud<pcl::PointXY>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*closest_waypoint, *temp_cloud);

    for (const auto& point : temp_cloud->points) {
        pcl::PointXY xy_point;
        xy_point.x = point.x;
        xy_point.y = point.y;
        cloud->points.push_back(xy_point);
    }

    pcl::KdTreeFLANN<pcl::PointXY> kdtree;
    kdtree.setInputCloud(cloud);

    for (const auto& box : input_bbox_array.boxes) {
        pcl::PointXY search_point;
        search_point.x = box.pose.position.x;
        search_point.y = box.pose.position.y;
        
        std::vector<int> point_indices;
        std::vector<float> point_distances;
        if (kdtree.radiusSearch(search_point, radius, point_indices, point_distances) > 0) {
            output_bbox_array.boxes.push_back(box);
        }
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    time_taken = elapsed_seconds.count();
    // saveTimeToFile(crophdmap_time_log_path, time_taken);
}

void Tracking::tracking(const jsk_recognition_msgs::BoundingBoxArray &bbox_array, 
                        jsk_recognition_msgs::BoundingBoxArray &track_bbox_array, 
                        visualization_msgs::MarkerArray &track_text_array, std::vector<bool> &deep_check_array, 
                        const double &egoVehicle_yaw,
                        const ros::Time &input_stamp, double& time_taken)
{
    auto start = std::chrono::steady_clock::now();

    track_bbox_array.boxes.clear();
    track_bbox_array.header.stamp = input_stamp;
    track_text_array.markers.clear();
    tracker.predictNewLocationOfTracks(input_stamp);    // Prediction
    tracker.assignDetectionsTracks(bbox_array);         // Matching
    tracker.assignedTracksUpdate(bbox_array);    
    tracker.unassignedTracksUpdate();
    tracker.deleteLostTracks();
    tracker.createNewTracks(bbox_array, egoVehicle_yaw);
    auto bbox = tracker.displayTrack();
    track_bbox_array = bbox.first;    
    track_text_array = bbox.second;

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    time_taken = elapsed_seconds.count();
    // saveTimeToFile(tracking_time_log_path, time_taken);
}

void Tracking::transformBbox(const jsk_recognition_msgs::BoundingBoxArray &input_bbox_array, tf2_ros::Buffer &tf_buffer, 
                            jsk_recognition_msgs::BoundingBoxArray &output_bbox_array, double& time_taken) 
{
    auto start = std::chrono::high_resolution_clock::now();

    output_bbox_array.boxes.clear();

    geometry_msgs::TransformStamped transformStamped;
    try {
        transformStamped = tf_buffer.lookupTransform(target_frame, lidar_frame, ros::Time(0)); // static tf
    } catch (tf2::TransformException &ex) {
        output_bbox_array = input_bbox_array;
        return;
    }

    for (const auto &box : input_bbox_array.boxes) {
        geometry_msgs::PoseStamped input_pose, output_pose;

        input_pose.pose = box.pose;
        tf2::doTransform(input_pose, output_pose, transformStamped);

        jsk_recognition_msgs::BoundingBox transformed_box;
        transformed_box.header = box.header;
        transformed_box.header.frame_id = target_frame;
        transformed_box.pose = output_pose.pose;
        transformed_box.dimensions = box.dimensions;
        transformed_box.value = box.value;
        transformed_box.label = box.label;
        output_bbox_array.boxes.push_back(transformed_box);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    time_taken = elapsed_seconds.count();
    // saveTimeToFile(transform_time_log_path, time_taken);
}

void Tracking::correctionBboxRelativeSpeed(const jsk_recognition_msgs::BoundingBoxArray &input_bbox_array, const ros::Time &input_stamp, 
                            const ros::Time &cur_stamp, jsk_recognition_msgs::BoundingBoxArray &output_bbox_array, double& time_taken) 
{
    auto start = std::chrono::steady_clock::now();

    output_bbox_array.boxes.clear();

    double delta_time = (cur_stamp - input_stamp).toSec();

    for (const auto &box : input_bbox_array.boxes) {
        jsk_recognition_msgs::BoundingBox corrected_box = box; // 원래 box 복사
        corrected_box.header.stamp = cur_stamp;

        if (corrected_box.header.seq > invisibleCnt / 2 && corrected_box.label == 1) {
            
            double velocity = 0.704525;
            double yaw = tf::getYaw(box.pose.orientation);
            double delta_x = velocity * 0.2 * cos(yaw);
            double delta_y = velocity * 0.2 * sin(yaw);
            // 100km/h & 0.1sec -> 2.76m 
            delta_x = std::copysign(std::min(std::abs(delta_x), 2.8), delta_x);
            delta_y = std::copysign(std::min(std::abs(delta_y), 2.8), delta_y);

            corrected_box.pose.position.x += delta_x; // x 방향으로 이동
            corrected_box.pose.position.y += delta_y; // y 방향으로 이동                    
        }
        
        output_bbox_array.boxes.push_back(corrected_box);
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    time_taken = elapsed_seconds.count();
    // saveTimeToFile(correction_time_log_path, time_taken);
}

void Tracking::postProcessing(const jsk_recognition_msgs::BoundingBoxArray &corrected_bbox_array, lidar_tracking::AdjacentVehicle &msg_PostProcessing, const boost::shared_ptr<EgoLocalization>& st_EgoInfo, double& time_taken)
{
    ros::Time start_time = ros::Time::now();

    for (int i = 0; i < 8; ++i)
        msg_PostProcessing.region_flags[i] = false;

    msg_PostProcessing.ar_PoseVehicles.poses.clear();
    msg_PostProcessing.ar_PoseVehicles.header.stamp = ros::Time::now();
    // msg_PostProcessing.ar_PoseVehicles.header.frame_id = "base_link";

    for (const auto& bbox : corrected_bbox_array.boxes) {
        float x = bbox.pose.position.x;
        float y = bbox.pose.position.y;

        // msg_PostProcessing.ar_PoseVehicles.poses.push_back(bbox.pose);

        int region = 0;
        if (15 <= x && x <= 30 && 0 < y && y <= 4) region = 1;
        else if (0 <= x && x < 15 && 0 < y && y <= 4) region = 2;
        else if (-15 <= x && x < 0 && 0 < y && y <= 4) region = 3;
        else if (-30 <= x && x < -15 && 0 < y && y <= 4) region = 4;
        else if (15 <= x && x <= 30 && -4 <= y && y < 0) region = 5;
        else if (0 <= x && x < 15 && -4 <= y && y < 0) region = 6;
        else if (-15 <= x && x < 0 && -4 <= y && y < 0) region = 7;
        else if (-30 <= x && x < -15 && -4 <= y && y < 0) region = 8;

        // std::cout << "\033[33mRelative Velocity: " << int(bbox.value * 3.6) << "\033[0m" << std::endl;

        // 유효 거리 내에 객체에 대한 3초 예측
        if(region==2 || region==3 || region==6 || region==7)
        {
            geometry_msgs::Pose pose_with_velocity = bbox.pose;

            // 1. Ego 속도
            double v_ego = st_EgoInfo->v;  // 자차 속력 (m/s)

            // 2. 객체 상대 속도
            double v_rel = bbox.value;  // 상대 속도 (m/s)

            // 3. 객체 절대 속도 = 자차 속도 + 상대 속도
            double v_target = v_ego + v_rel;  // m/s

            // 4. 객체 Heading 방향
            double theta_target = tf::getYaw(bbox.pose.orientation);

            // 5. 절대 속도를 객체 heading 방향에 적용한 속도 벡터
            double vx_target = v_target * std::cos(theta_target);
            double vy_target = v_target * std::sin(theta_target);

            // 6. 현재 위치에 속력 정보를 z에 저장
            pose_with_velocity.position.z = v_target * 3.6;  // km/h 저장

            // 7. 현재 위치 저장
            msg_PostProcessing.ar_PoseVehicles.poses.push_back(pose_with_velocity);

            // 8. 3초 후 예측 위치 계산
            geometry_msgs::Pose future_pose;
            future_pose.position.x = bbox.pose.position.x + vx_target * thresh_predictSec;
            future_pose.position.y = bbox.pose.position.y + vy_target * thresh_predictSec;
            future_pose.position.z = v_target * 3.6;  // 동일한 속도 정보

            future_pose.orientation = bbox.pose.orientation;

            // 9. 예측 위치 저장
            msg_PostProcessing.ar_PoseVehicles.poses.push_back(future_pose);
            
            // [ Ego 고려 ]
            // geometry_msgs::Pose pose_with_velocity = bbox.pose;

            // // 1. 자차 속도 벡터
            // double v_ego = st_EgoInfo->v;
            // double theta_ego = st_EgoInfo->yaw;
            // double vx_ego = v_ego * std::cos(theta_ego);
            // double vy_ego = v_ego * std::sin(theta_ego);

            // // 2. 상대 속도 벡터 (bbox.value 사용 but 방향은 bbox.pose.orientation 사용)
            // double v_rel = bbox.value;  // m/s
            // double theta_target = tf::getYaw(bbox.pose.orientation);
            // double vx_rel = v_rel * std::cos(theta_target);
            // double vy_rel = v_rel * std::sin(theta_target);

            // // 3. 절대 속도 벡터
            // double vx_target = vx_ego + vx_rel;
            // double vy_target = vy_ego + vy_rel;

            // // 4. 절대 속력 저장 (for z)
            // double v_target = std::sqrt(vx_target * vx_target + vy_target * vy_target);
            // pose_with_velocity.position.z = v_target * 3.6;  // km/h 저장

            // // 5. 현재 위치 저장
            // msg_PostProcessing.ar_PoseVehicles.poses.push_back(pose_with_velocity);

            // // 6. 예측 위치 계산
            // geometry_msgs::Pose future_pose;
            // future_pose.position.x = bbox.pose.position.x + vx_target * thresh_predictSec;
            // future_pose.position.y = bbox.pose.position.y + vy_target * thresh_predictSec;
            // future_pose.position.z = v_target * 3.6;

            // future_pose.orientation = bbox.pose.orientation;

            // // 7. 예측 위치 저장
            // msg_PostProcessing.ar_PoseVehicles.poses.push_back(future_pose);
        }

        if (region > 0)
            msg_PostProcessing.region_flags[region - 1] = true;
    }

    // // Check Result
    // std::cout << "\033[31m[PostProcessing]\033[0m Region Flags: ";
    // for (int i = 0; i < 8; ++i) {
    //     std::cout << "R" << (i + 1) << ": ";
    //     std::cout << "\033[31m" << (msg_PostProcessing.region_flags[i] ? "true" : "false") << "\033[0m  ";
    // }
    // std::cout << std::endl;

    // Check Predict Position Result
    std::cout << "\033[34m--- [AdjacentVehicle Pose Info] ---\033[0m" << std::endl;

    const auto& poses = msg_PostProcessing.ar_PoseVehicles.poses;

    for (size_t i = 0; i < poses.size(); ++i) {
        const auto& p = poses[i];
        std::cout << "\033[34m[" << (i % 2 == 0 ? "Current" : "Future") << "] "
                << "x: " << p.position.x << ", "
                << "y: " << p.position.y << ", "
                << "speed(km/h): " << p.position.z << "\033[0m" << std::endl;
    }

    msg_PostProcessing.s32_NumVehicles = msg_PostProcessing.ar_PoseVehicles.poses.size()/2;

    ros::Duration duration = ros::Time::now() - start_time;
    time_taken = duration.toSec();
}

void Tracking::correctionBboxTF(const jsk_recognition_msgs::BoundingBoxArray &input_bbox_array, const ros::Time &input_stamp, 
                                const ros::Time &cur_stamp, tf2_ros::Buffer &tf_buffer, 
                                jsk_recognition_msgs::BoundingBoxArray &output_bbox_array, double& time_taken) 
{
    auto start = std::chrono::steady_clock::now();

    output_bbox_array.boxes.clear();

    geometry_msgs::TransformStamped transformStampedAtInput, transformStampedAtCur;

    try {
        // 로봇 좌표계에서 월드 좌표계로의 변환을 가져옵니다.
        transformStampedAtInput = tf_buffer.lookupTransform(world_frame, target_frame, input_stamp);
        transformStampedAtCur = tf_buffer.lookupTransform(world_frame, target_frame, cur_stamp);
    } catch (tf2::TransformException &ex) {
        output_bbox_array = input_bbox_array;
        return;
    }

    tf2::Transform tfAtInput, tfAtCur, deltaTransform;
    tf2::fromMsg(transformStampedAtInput.transform, tfAtInput);
    tf2::fromMsg(transformStampedAtCur.transform, tfAtCur);

    // 두 시점 간의 로봇의 움직임을 계산합니다.
    deltaTransform = tfAtCur.inverse() * tfAtInput;

    // deltaTransform을 geometry_msgs::TransformStamped로 변환합니다.
    geometry_msgs::TransformStamped deltaTransformStamped;
    deltaTransformStamped.header.stamp = input_stamp;
    deltaTransformStamped.header.frame_id = target_frame;
    deltaTransformStamped.child_frame_id = target_frame;
    deltaTransformStamped.transform = tf2::toMsg(deltaTransform);

    for (const auto &box : input_bbox_array.boxes) {
        geometry_msgs::PoseStamped input_pose, transformed_pose;

        input_pose.header = box.header;
        input_pose.pose = box.pose;

        // deltaTransform을 바운딩 박스에 적용합니다.
        tf2::doTransform(input_pose, transformed_pose, deltaTransformStamped);

        jsk_recognition_msgs::BoundingBox transformed_box;
        transformed_box.header = box.header;
        transformed_box.header.stamp = cur_stamp; // 보정된 시점으로 업데이트
        transformed_box.pose = transformed_pose.pose;
        transformed_box.dimensions = box.dimensions;
        transformed_box.value = box.value;
        transformed_box.label = box.label;
        output_bbox_array.boxes.push_back(transformed_box);
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    time_taken = elapsed_seconds.count();
    // saveTimeToFile(correction_time_log_path, time_taken);
}

