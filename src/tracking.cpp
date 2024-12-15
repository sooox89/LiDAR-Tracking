#include <iostream>
#include <csignal>
#include "tracking/tracking.hpp"
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

ros::Publisher pub_track_box, pub_track_text, pub_track_model, pub_track_test;

// Track tracker;
boost::shared_ptr<Tracking> Tracking_;  // Tracking 클래스의 객체를 boost::shared_ptr로 관리

double t9, t10, t11, t12, t13, total;
std::string fixed_frame;

tf2_ros::Buffer tf_buffer;

jsk_recognition_msgs::BoundingBoxArray cluster_bbox_array, deep_bbox_array, integration_bbox_array, filtered_bbox_array, 
                                        output_bbox_array, track_bbox_array, transformed_bbox_array, corrected_bbox_array;
visualization_msgs::MarkerArray track_text_array, track_model_array;

std::string lidar_frame, target_frame, world_frame;

ros::Time publish_stamp;

// ROI 사각형의 좌표
const float ROI_X_MIN = -20.0f;
const float ROI_X_MAX = 20.0f;
const float ROI_Y_MIN = -15.0f;
const float ROI_Y_MAX = 15.0f;

// ROI 필터링 함수
void filterROI(const jsk_recognition_msgs::BoundingBoxArray& input_bbox_array,
              jsk_recognition_msgs::BoundingBoxArray& filtered_bbox_array) {

    filtered_bbox_array.header = input_bbox_array.header;
    filtered_bbox_array.boxes.clear();

    for (const auto& box : input_bbox_array.boxes) {
        // 박스의 중심 좌표
        float x = box.pose.position.x;
        float y = box.pose.position.y;
        
        // ROI 내에 포함되는 박스만 필터링
        if (x >= ROI_X_MIN && x <= ROI_X_MAX && y >= ROI_Y_MIN && y <= ROI_Y_MAX) {
            filtered_bbox_array.boxes.push_back(box);
        }
    }
}

// 동기화된 콜백 함수 수정
void callbackSynchronized(const jsk_recognition_msgs::BoundingBoxArray::ConstPtr &cluster_bba_msg,
                          const jsk_recognition_msgs::BoundingBoxArray::ConstPtr &deep_bba_msg)
{
    cluster_bbox_array = *cluster_bba_msg;
    deep_bbox_array = *deep_bba_msg;

    // ROI 필터링 적용
    jsk_recognition_msgs::BoundingBoxArray cluster_filtered_bbox_array;
    jsk_recognition_msgs::BoundingBoxArray deep_filtered_bbox_array;
    filterROI(cluster_bbox_array, cluster_filtered_bbox_array);
    filterROI(deep_bbox_array, deep_filtered_bbox_array);

    // 필터링된 데이터를 통합
    Tracking_->integrationBbox(cluster_filtered_bbox_array, deep_filtered_bbox_array, integration_bbox_array, t9);

    if (checkTransform(tf_buffer, world_frame, target_frame)) {
        Tracking_->transformBbox(integration_bbox_array, tf_buffer, transformed_bbox_array, t10);
        fixed_frame = target_frame;
        output_bbox_array = transformed_bbox_array;
    } else {
        fixed_frame = lidar_frame;
        output_bbox_array = integration_bbox_array;
    }

    Tracking_->tracking(output_bbox_array, track_bbox_array, track_text_array, cluster_bba_msg->header.stamp, t12);
    Tracking_->correctionBboxRelativeSpeed(track_bbox_array, cluster_bba_msg->header.stamp, publish_stamp, corrected_bbox_array, t13);

    publish_stamp = ros::Time::now();

    pub_track_box.publish(bba2msg(track_bbox_array, publish_stamp, fixed_frame));
    pub_track_model.publish(bba2ma(track_bbox_array, publish_stamp, fixed_frame));
    pub_track_text.publish(ta2msg(track_text_array, publish_stamp, fixed_frame));

    total = ros::Time::now().toSec() - cluster_bba_msg->header.stamp.toSec();

    std::cout << "\033[" << 18 << ";" << 30 << "H" << std::endl;
    std::cout << "integration & transform : " << t9 + t10 << " sec" << std::endl;
    std::cout << "tracking : " << t12 << " sec" << std::endl;
    std::cout << "total : " << total << " sec" << std::endl;
    std::cout << "fixed frame : " << fixed_frame << std::endl;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "tracking");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    tf2_ros::TransformListener tf_listener(tf_buffer);

    pnh.param<std::string>("lidar_frame", lidar_frame, "hesai_lidar");
    pnh.param<std::string>("target_frame", target_frame, "ego_car");
    pnh.param<std::string>("world_frame", world_frame, "world");

    pub_track_box = pnh.advertise<jsk_recognition_msgs::BoundingBoxArray>("/mobinha/perception/lidar/track_box", 10);
    pub_track_text = pnh.advertise<visualization_msgs::MarkerArray>("/mobinha/visualize/visualize/track_text", 10);
    pub_track_model = pnh.advertise<visualization_msgs::MarkerArray>("/mobinha/visualize/visualize/track_model", 10);

    // Tracking 객체를 초기화
    Tracking_ = boost::make_shared<Tracking>(pnh);

    // message_filters를 사용하여 토픽 구독 및 동기화 설정
    message_filters::Subscriber<jsk_recognition_msgs::BoundingBoxArray> sub_cluster_box(nh, "/cloud_segmentation/cluster_box", 10);
    message_filters::Subscriber<jsk_recognition_msgs::BoundingBoxArray> sub_deep_box(nh, "/deep_box", 10);

    typedef message_filters::sync_policies::ApproximateTime<jsk_recognition_msgs::BoundingBoxArray, jsk_recognition_msgs::BoundingBoxArray> SyncPolicy;
    message_filters::Synchronizer<SyncPolicy> sync(SyncPolicy(10), sub_cluster_box, sub_deep_box);
    sync.registerCallback(boost::bind(&callbackSynchronized, _1, _2));

    // Waypoints 구독
    ros::Subscriber sub_waypoints = nh.subscribe("/waypoints", 1, &Tracking::updateWaypoints, Tracking_.get());

    ros::spin();
    return 0;
}