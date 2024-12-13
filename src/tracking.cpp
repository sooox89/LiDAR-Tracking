#include <ros/ros.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <visualization_msgs/MarkerArray.h>
#include <std_msgs/Bool.h>
#include <cmath>
#include <vector>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

// 퍼블리셔 선언
ros::Publisher pub_track_box;
ros::Publisher pub_hazard_warning;
ros::Publisher pub_target_circles;

// 프레임 및 파라미터
std::string lidar_frame, target_frame, world_frame;
std::vector<std::pair<float, float>> target_points;
float radius;

// ROI 필터링 함수
void filterROI(const jsk_recognition_msgs::BoundingBoxArray& input_bbox_array,
               jsk_recognition_msgs::BoundingBoxArray& filtered_bbox_array) {
    const float x_min = -20.0f, x_max = 20.0f, y_min = -15.0f, y_max = 15.0f;

    filtered_bbox_array.header = input_bbox_array.header;
    filtered_bbox_array.boxes.clear();

    for (const auto& box : input_bbox_array.boxes) {
        float x = box.pose.position.x, y = box.pose.position.y;
        if (x >= x_min && x <= x_max && y >= y_min && y <= y_max) {
            filtered_bbox_array.boxes.push_back(box);
        }
    }
}

// track_box 원 마커 중심 위험 판단
void evaluateHazards(const jsk_recognition_msgs::BoundingBoxArray& track_box,
                     const std::vector<std::pair<float, float>>& target_points,
                     float radius, std_msgs::Bool& hazard_msg) {
    hazard_msg.data = false;

    for (const auto& box : track_box.boxes) {
        float x = box.pose.position.x, y = box.pose.position.y;

        for (const auto& target : target_points) {
            float dx = x - target.first, dy = y - target.second;
            if (std::sqrt(dx * dx + dy * dy) <= radius) {
                ROS_WARN("Hazard detected near target point: [%.2f, %.2f]", x, y);
                hazard_msg.data = true;
                return;
            }
        }
    }
}

// Target Points 원 시각화
void visualizeTargetPoints(const std::vector<std::pair<float, float>>& target_points, float radius) {
    visualization_msgs::MarkerArray marker_array;

    for (size_t i = 0; i < target_points.size(); ++i) {
        const auto& point = target_points[i];

        visualization_msgs::Marker circle_marker;
        circle_marker.header.frame_id = target_frame;
        circle_marker.header.stamp = ros::Time::now();
        circle_marker.ns = "target_circles";
        circle_marker.id = i;
        circle_marker.type = visualization_msgs::Marker::CYLINDER;
        circle_marker.action = visualization_msgs::Marker::ADD;

        circle_marker.pose.position.x = point.first;
        circle_marker.pose.position.y = point.second;
        circle_marker.pose.position.z = 0.0;

        circle_marker.scale.x = radius * 2.0; // 원 지름
        circle_marker.scale.y = radius * 2.0; // 원 지름
        circle_marker.scale.z = 0.1;          // 두께

        circle_marker.color.r = 0.0f;
        circle_marker.color.g = 1.0f;
        circle_marker.color.b = 0.0f;
        circle_marker.color.a = 0.5f; // 투명도

        marker_array.markers.push_back(circle_marker);
    }

    pub_target_circles.publish(marker_array);
}

// Target Points 업데이트 콜백
void targetPointsCallback(const visualization_msgs::MarkerArray::ConstPtr& msg) {
    target_points.clear();

    for (const auto& marker : msg->markers) {
        float x = marker.pose.position.x;
        float y = marker.pose.position.y;
        target_points.emplace_back(x, y);
    }

    // 업데이트된 Target Points 시각화
    visualizeTargetPoints(target_points, radius);
}

// Synchronized Callback
void synchronizedCallback(const jsk_recognition_msgs::BoundingBoxArray::ConstPtr& cluster_bba_msg,
                          const jsk_recognition_msgs::BoundingBoxArray::ConstPtr& deep_bba_msg) {
    // Process track_box
    jsk_recognition_msgs::BoundingBoxArray track_bbox_array = *cluster_bba_msg;

    std_msgs::Bool hazard_msg;
    evaluateHazards(track_bbox_array, target_points, radius, hazard_msg);
    pub_hazard_warning.publish(hazard_msg);

    // Process deep_box
    jsk_recognition_msgs::BoundingBoxArray deep_bbox_array = *deep_bba_msg;
    jsk_recognition_msgs::BoundingBoxArray filtered_deep_bbox_array;
    filterROI(deep_bbox_array, filtered_deep_bbox_array);

    for (const auto& box : filtered_deep_bbox_array.boxes) {
        float speed = box.value; // 속도 정보 포함 가정
        if (speed > 5.0f) { 
            ROS_INFO("Dynamic obstacle detected: speed %.2f m/s", speed);
            // 추가적인 위험 판단 및 퍼블리시 가능
        }
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "tracking_with_roi");
    ros::NodeHandle nh, pnh("~");

    // 파라미터 로드
    pnh.param("radius", radius, 2.0f);
    pnh.param<std::string>("lidar_frame", lidar_frame, "hesai_lidar");
    pnh.param<std::string>("target_frame", target_frame, "ego_car");
    pnh.param<std::string>("world_frame", world_frame, "world");

    // 퍼블리셔 초기화
    pub_track_box = pnh.advertise<jsk_recognition_msgs::BoundingBoxArray>("/mobinha/perception/lidar/track_box", 10);
    pub_hazard_warning = pnh.advertise<std_msgs::Bool>("/mobinha/hazard_warning", 10);
    pub_target_circles = pnh.advertise<visualization_msgs::MarkerArray>("/mobinha/circles_viz", 10);

    // 서브스크라이버 초기화
    message_filters::Subscriber<jsk_recognition_msgs::BoundingBoxArray> sub_cluster_box(nh, "/cloud_segmentation/cluster_box", 10);
    message_filters::Subscriber<jsk_recognition_msgs::BoundingBoxArray> sub_deep_box(nh, "/deep_box", 10);
    ros::Subscriber sub_target_points = nh.subscribe("/target_points", 10, targetPointsCallback);

    // 메시지 동기화 설정
    typedef message_filters::sync_policies::ApproximateTime<jsk_recognition_msgs::BoundingBoxArray, jsk_recognition_msgs::BoundingBoxArray> SyncPolicy;
    message_filters::Synchronizer<SyncPolicy> sync(SyncPolicy(10), sub_cluster_box, sub_deep_box);
    sync.registerCallback(synchronizedCallback);

    ros::spin();
    return 0;
}
