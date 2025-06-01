#include <iostream>
#include <csignal>
#include "tracking/tracking.hpp"

ros::Publisher pub_track_box, pub_track_text, pub_track_model, pub_track_test, pub_synchronized_cloud, pub_integration_box, pub_postProcessing, pub_adjacent_markers;

// Track tracker;
boost::shared_ptr<Tracking> Tracking_;
boost::shared_ptr<EgoLocalization> EgoLocalization_;

double t9, t10, t11, t12, t13, t14, total;
std::string fixed_frame;

tf2_ros::Buffer tf_buffer;

jsk_recognition_msgs::BoundingBoxArray cluster_bbox_array, deep_bbox_array, integration_bbox_array, filtered_bbox_array, 
                                        output_bbox_array, track_bbox_array, transformed_bbox_array, corrected_bbox_array;
visualization_msgs::MarkerArray track_text_array, track_model_array;

lidar_tracking::AdjacentVehicle msg_PostProcessing;

std::vector<bool> deep_check_array;    // Checklist for Deep Learning based Object 
std::string lidar_frame, target_frame, world_frame;

ros::Time publish_stamp;

void callbackSynchronized(const jsk_recognition_msgs::BoundingBoxArray::ConstPtr &cluster_bba_msg,
                          const jsk_recognition_msgs::BoundingBoxArray::ConstPtr &deep_bba_msg)
{
    cluster_bbox_array = *cluster_bba_msg;
    deep_bbox_array = *deep_bba_msg;

    Tracking_->integrationBbox(cluster_bbox_array, deep_bbox_array, integration_bbox_array, deep_check_array, t9);
    std::cout << "\033[" << 18 << ";" << 30 << "H" << std::endl;

    if (checkTransform(tf_buffer, world_frame, target_frame)) {
        Tracking_->transformBbox(integration_bbox_array, tf_buffer, transformed_bbox_array, t10);
        Tracking_->cropHDMapBbox(transformed_bbox_array, filtered_bbox_array, cluster_bba_msg->header.stamp, t11);
        fixed_frame = target_frame;
        output_bbox_array = filtered_bbox_array;
    } else {
        fixed_frame = lidar_frame;
        output_bbox_array = integration_bbox_array;
    }

    publish_stamp = ros::Time::now();

    Tracking_->tracking(output_bbox_array, track_bbox_array, track_text_array, deep_check_array, EgoLocalization_->yaw, cluster_bba_msg->header.stamp, t12);
    Tracking_->correctionBboxRelativeSpeed(track_bbox_array, cluster_bba_msg->header.stamp, publish_stamp, corrected_bbox_array, t13);

    // Tracking Object Post-processing
    // x_dist, y_dist Parameter Define
    Tracking_->postProcessing(corrected_bbox_array, msg_PostProcessing, EgoLocalization_, t14);
    pub_postProcessing.publish(msg_PostProcessing);

    // Pub Point
    // Marker 메시지 생성
    visualization_msgs::Marker marker;
    marker.header.frame_id = fixed_frame;
    marker.header.stamp = publish_stamp;
    marker.ns = "adjacent_vehicle_points";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::SPHERE_LIST;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 1.5;
    marker.scale.y = 1.5;
    marker.scale.z = 1.5;
    marker.lifetime = ros::Duration(0.3);  // 지속시간

    // 파란색 = 현재 위치, 빨간색 = 예측 위치
    for (size_t i = 0; i < msg_PostProcessing.ar_PoseVehicles.poses.size(); ++i) {
        const auto& pose = msg_PostProcessing.ar_PoseVehicles.poses[i];
        geometry_msgs::Point p;
        p.x = pose.position.x;
        p.y = pose.position.y;
        p.z = 0.3;
        marker.points.push_back(p);

        std_msgs::ColorRGBA color;
        if (i % 2 == 0) {  // 현재 위치
            color.r = 0.0; color.g = 0.0; color.b = 1.0; color.a = 1.0;
        } else {           // 예측 위치
            color.r = 1.0; color.g = 0.0; color.b = 0.0; color.a = 1.0;
        }
        marker.colors.push_back(color);
    }

    pub_adjacent_markers.publish(marker);

    pub_track_box.publish(bba2msg(corrected_bbox_array, publish_stamp, fixed_frame));
    pub_track_model.publish(bba2ma(corrected_bbox_array, publish_stamp, fixed_frame));
    pub_track_text.publish(ta2msg(track_text_array, publish_stamp, fixed_frame));

    total = ros::Time::now().toSec() - cluster_bba_msg->header.stamp.toSec();

    std::cout << "integration & transform : " << t9 + t10 << " sec" << std::endl;
    std::cout << "tracking : " << t12 << " sec" << std::endl;
    std::cout << "total : " << total << " sec" << std::endl;
    std::cout << "fixed frame : " << fixed_frame << std::endl;
}

void callbackCloud(const sensor_msgs::PointCloud2::Ptr &cloud_msg)
{
    if (cloud_msg->data.empty()) return;

    sensor_msgs::PointCloud2 cloud = *cloud_msg;
    cloud.header.stamp = ros::Time::now();

    pub_synchronized_cloud.publish(cloud);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "tracking");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    tf2_ros::TransformListener tf_listener(tf_buffer);

    // frame 설정
    pnh.param<std::string>("lidar_frame", lidar_frame, "hesai_lidar");
    pnh.param<std::string>("target_frame", target_frame, "ego_car");
    pnh.param<std::string>("world_frame", world_frame, "world");

    pub_track_box = pnh.advertise<jsk_recognition_msgs::BoundingBoxArray>("/mobinha/perception/lidar/track_box", 1);
    pub_track_text = pnh.advertise<visualization_msgs::MarkerArray>("/mobinha/visualize/visualize/track_text", 1);
    pub_track_model = pnh.advertise<visualization_msgs::MarkerArray>("/mobinha/visualize/visualize/track_model", 1);
    pub_synchronized_cloud = pnh.advertise<sensor_msgs::PointCloud2>("/cloud_segmentation/synchronized_cloud", 1);

    // // Post-processing Publisher
    pub_postProcessing = pnh.advertise<lidar_tracking::AdjacentVehicle>("/lidar_tracking/adjacent_vehicles", 1);
    pub_adjacent_markers = pnh.advertise<visualization_msgs::Marker>("/lidar_tracking/adjacent_vehicle_markers", 1);

    // pub_integration_box = pnh.advertise<jsk_recognition_msgs::BoundingBoxArray>("/mobinha/perception/lidar/integration_box", 1);

    Tracking_ = boost::make_shared<Tracking>(pnh);
    EgoLocalization_ = boost::make_shared<EgoLocalization>();

    // Ego Vehicle Orientation
    ros::Subscriber sub_inspva = nh.subscribe("/novatel/oem7/inspva", 1, &EgoLocalization::update, EgoLocalization_.get());

    // message_filters
    message_filters::Subscriber<jsk_recognition_msgs::BoundingBoxArray> sub_cluster_box(nh, "/cloud_segmentation/cluster_box", 1);
    message_filters::Subscriber<jsk_recognition_msgs::BoundingBoxArray> sub_deep_box(nh, "/deep_box", 1);

    typedef message_filters::sync_policies::ApproximateTime<jsk_recognition_msgs::BoundingBoxArray, jsk_recognition_msgs::BoundingBoxArray> SyncPolicy;
    message_filters::Synchronizer<SyncPolicy> sync(SyncPolicy(10), sub_cluster_box, sub_deep_box);
    sync.setMaxIntervalDuration(ros::Duration(0.05));
    sync.registerCallback(boost::bind(&callbackSynchronized, _1, _2));

    ros::Subscriber sub_waypoints = nh.subscribe("/waypoints", 1, &Tracking::updateWaypoints, Tracking_.get());
    ros::Subscriber sub_cloud = nh.subscribe("/cloud_segmentation/undistortioncloud", 1, callbackCloud);

    ros::spin();
    return 0;
}