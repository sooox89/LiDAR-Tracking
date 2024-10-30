#include <iostream>
#include <csignal>
#include "tracking/tracking.hpp"

ros::Publisher pub_track_box;
ros::Publisher pub_track_text;
ros::Publisher pub_track_model;

// Track tracker;
boost::shared_ptr<Tracking> Tracking_;  // Tracking 클래스의 객체를 boost::shared_ptr로 관리

// 처리 시간 변수 : 추적 과정 소요 시간 기록
double t9, t10, t11, t12, t13, total;
std::string fixed_frame;

// 좌표 변환
tf2_ros::Buffer tf_buffer;
// bounding box 배열 
jsk_recognition_msgs::BoundingBoxArray cluster_bbox_array, deep_bbox_array, integration_bbox_array, filtered_bbox_array, track_bbox_array, transformed_bbox_array, corrected_bbox_array, output_bbox_array;
visualization_msgs::MarkerArray track_text_array, track_model_array;

// 프레임 이름
std::string lidar_frame, target_frame, world_frame;

// 시그널 처리 함수 : Tracking 평균 처리 시간 출력
void signalHandler(int signum) {
    if (Tracking_) {
        Tracking_->averageTime();  // 프로그램 종료 전에 averageTime 호출
    }
    exit(signum); // 프로그램 종료
}

// 클러스터링 결과 처리 콜백 함수 : 클러스터링 된 bounding box 데이터 수신히여 처리
void callbackCluster(const jsk_recognition_msgs::BoundingBoxArray::Ptr &bba_msg)
{   
    if (bba_msg->boxes.empty()) { return; }

    cluster_bbox_array = *bba_msg;
    // 클러스터링된 b.box & 딥러닝 기반 b.box 통합
    Tracking_->integrationBbox(cluster_bbox_array, deep_bbox_array, integration_bbox_array, t9);
    // 통합된 b.box -> HD 지도 정보 기반으로 crop
    Tracking_->cropHDMapBbox(integration_bbox_array, filtered_bbox_array, bba_msg->header.stamp, tf_buffer, target_frame, world_frame, t10);
    // 필터링된 b.box 사용해서 tracking 수행
    Tracking_->tracking(filtered_bbox_array, track_bbox_array, track_text_array, bba_msg->header.stamp, t11);
    // 속도 정보 기반으로 추적된 b.box 보정
    // Tracking_->correctionBboxRelativeSpeed(track_bbox_array, bba_msg->header.stamp, ros::Time::now(), corrected_bbox_array, t12);
    
    // LiDAR 프레임과 대상 프레임간의 변환 가능한지 확인 -> 가능하면 좌표 변환 수행
    // 최종적으로 변환된 b.box 사용해서 출력 준비
    if (checkTransform(tf_buffer, lidar_frame, target_frame)) {
        // corrected_bbox_array를 lidar_frame -> target_frame 변환
        // 변환 결과 transformed_bbox_array에 저장
        Tracking_->transformBbox(track_bbox_array, lidar_frame, target_frame, tf_buffer, transformed_bbox_array, t13);
        // Tracking_->correctionBbox(transformed_bbox_array, bba_msg->header.stamp, ros::Time::now(), target_frame, world_frame, tf_buffer, corrected_bbox_array, t13);
        // 변환 성공 시, 최종적으로 데이터 target_frame 기준으로 해석
        fixed_frame = target_frame;  //fixed_frame : 데이터 퍼블리싱, 시각화 할 때 사용할 기준 프레임 설정
        // 변환된 b.box 배열을 출력으로 사용
        output_bbox_array = transformed_bbox_array;
    } else {
    // TF 정보가 없는 상황에서 최소한 LiDAR 데이터 자체 사용가능하게 
        fixed_frame = lidar_frame;
        output_bbox_array = track_bbox_array;
    }
    
    // 데이터 퍼블리싱 : 추적 결과 publish
    pub_track_box.publish(bba2msg(output_bbox_array, ros::Time::now(), fixed_frame));
    pub_track_model.publish(bba2ma(output_bbox_array, ros::Time::now(), fixed_frame));
    pub_track_text.publish(ta2msg(track_text_array, ros::Time::now(), fixed_frame));
    
    total = ros::Time::now().toSec() - cluster_bbox_array.boxes[0].header.stamp.toSec();

    std::cout << "\033[" << 18 << ";" << 30 << "H" << std::endl;
    std::cout << "integration & crophdmap : " << t9+t10 << "sec" << std::endl;
    std::cout << "tracking : " << t11 << "sec" << std::endl;
    std::cout << "correction : " << t12 << "sec" << std::endl;
    std::cout << "transform : " << t13 << "sec" << std::endl;
    std::cout << "total : " << total << " sec" << std::endl;
    std::cout << "fixed frame : " << fixed_frame << std::endl;
}

// 딥러닝 결과 처리 콜백 함수 : deep_bbox_array에 저장해서 클러스터링 데이터와 통합할 때 사용
void callbackDeep(const jsk_recognition_msgs::BoundingBoxArray::Ptr &bba_msg)
{
    if (bba_msg->boxes.empty()) { return; }

    deep_bbox_array = *bba_msg;
}


// Main 함수 : 
int main(int argc, char** argv)
{
    ros::init(argc, argv, "tracking"); 
    ros::NodeHandle nh;  // ros 매개변수 가져옴
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

    ros::Subscriber sub_cluster_box = nh.subscribe("/cloud_segmentation/cluster_box", 10, callbackCluster);
    ros::Subscriber sub_deep_box = nh.subscribe("/deep_box", 10, callbackDeep);

    signal(SIGINT, signalHandler);  

    ros::spin();
    return 0;
}



