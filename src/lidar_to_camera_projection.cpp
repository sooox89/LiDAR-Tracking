// lidar_to_camera_projection.cpp
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/PointCloud2.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <pcl_ros/transforms.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <image_transport/image_transport.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <pcl/filters/voxel_grid.h>

// 포인트 타입 정의
typedef pcl::PointXYZI PointType;
typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::CompressedImage, sensor_msgs::PointCloud2> SyncPolicy;
typedef message_filters::Synchronizer<SyncPolicy> Synchronizer;

class LidarToCameraProjection {
public:
    LidarToCameraProjection(ros::NodeHandle& nh, ros::NodeHandle& pnh) 
        : tf_listener_(tf_buffer_), it_(nh)
    {
        // 프레임 이름 및 토픽 파라미터 로드
        pnh.param<std::string>("lidar_topic", lidar_topic_, "/lidar_points");
        pnh.param<std::string>("image_topic", image_topic_, "/gmsl_camera/dev/video0/compressed");
        pnh.param<std::string>("lidar_frame", lidar_frame_, "hesai_lidar");
        pnh.param<std::string>("camera_frame", camera_frame_, "gmsl_camera");
        pnh.param<std::string>("world_frame", world_frame_, "world");
        pnh.param<std::string>("ego_frame", ego_frame_, "ego_car");

        // 정적 변환 퍼블리시
        publishStaticTransforms();

        // 메시지 필터를 사용하여 이미지와 포인트 클라우드 동기화
        image_sub_.subscribe(nh, image_topic_, 1);
        cloud_sub_.subscribe(nh, lidar_topic_, 1);
        
        sync_.reset(new Synchronizer(SyncPolicy(30), image_sub_, cloud_sub_));
        sync_->registerCallback(boost::bind(&LidarToCameraProjection::syncedCallback, this, _1, _2));

        // image_transport을 사용하여 오버레이 이미지 퍼블리시
        overlay_image_pub_ = it_.advertise("/camera/lidar_overlay", 1);

        // 카메라 내재 파라미터 설정 (실제 카메라에 맞게 수정 필요)
        camera_matrix_ = (cv::Mat_<double>(3, 3) << 
            2013.07901, 0, 1003.25058, 
            0, 2006.52160, 593.024593, 
            0, 0, 1);
        dist_coeffs_ = (cv::Mat_<double>(5, 1) << 
            -0.37001482, 0.4684665, -0.00749647, -0.00681344, -0.68205982);

        ROS_INFO("Lidar to Camera Projection node initialized.");
    }

private:
    // 메시지 필터용 서브스크라이버
    typedef message_filters::Subscriber<sensor_msgs::CompressedImage> ImageSubscriber;
    typedef message_filters::Subscriber<sensor_msgs::PointCloud2> CloudSubscriber;
    boost::shared_ptr<Synchronizer> sync_;
    ImageSubscriber image_sub_;
    CloudSubscriber cloud_sub_;

    // image_transport을 사용한 퍼블리셔
    image_transport::ImageTransport it_;
    image_transport::Publisher overlay_image_pub_;

    // 현재 카메라 이미지 및 카메라 파라미터
    cv::Mat current_image_;
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    std::string lidar_topic_;
    std::string image_topic_;
    std::string lidar_frame_;
    std::string camera_frame_;
    std::string world_frame_;
    std::string ego_frame_;

    // TF 리스너 및 브로드캐스터
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    tf2_ros::StaticTransformBroadcaster static_broadcaster_;

    // 동기화된 콜백 함수
    void syncedCallback(const sensor_msgs::CompressedImageConstPtr& img_msg,
                        const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
    {
        ROS_DEBUG("syncedCallback triggered.");

        // PointCloud2 메시지의 frame_id가 비어 있는지 확인
        if (cloud_msg->header.frame_id.empty()) {
            ROS_ERROR("PointCloud2 message has an empty frame_id.");
            return;
        }

        // 압축된 이미지 디코딩
        try {
            // 디코드된 이미지의 형식을 명확히 해야 합니다. 예: "bgr8"
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(img_msg, "bgr8");
            current_image_ = cv_ptr->image;
            if (current_image_.empty()) {
                ROS_ERROR("Failed to decode compressed image.");
                return;
            }
        } catch (const cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        // 포인트 클라우드 메시지를 PCL 형식으로 변환
        pcl::PointCloud<PointType> cloud;
        pcl::fromROSMsg(*cloud_msg, cloud);
        ROS_DEBUG("PointCloud received with %zu points.", cloud.points.size());

        // 포인트 클라우드를 카메라 프레임으로 변환
        pcl::PointCloud<PointType> cloud_cam;
        if (!transformPointCloud(cloud, cloud_cam, camera_frame_, cloud_msg->header.frame_id, cloud_msg->header.stamp)) {
            ROS_WARN("Failed to transform point cloud to camera frame.");
            return;
        }
        ROS_DEBUG("Transformed PointCloud has %zu points.", cloud_cam.points.size());

        // 변환된 포인트 클라우드를 이미지에 오버레이
        cv::Mat overlay_image = current_image_.clone();
        size_t points_projected = 0;

        for (const auto& point : cloud_cam.points) {
            if (point.z <= 0) continue;

            double u = (camera_matrix_.at<double>(0, 0) * point.x / point.z) + camera_matrix_.at<double>(0, 2);
            double v = (camera_matrix_.at<double>(1, 1) * point.y / point.z) + camera_matrix_.at<double>(1, 2);
            int x = static_cast<int>(u);
            int y = static_cast<int>(v);

            // 로그로 프로젝트된 좌표 출력
            if (x >= 0 && x < overlay_image.cols && y >= 0 && y < overlay_image.rows) {
                ROS_DEBUG("Projected Point: x = %d, y = %d", x, y);
                cv::circle(overlay_image, cv::Point(x, y), 3, cv::Scalar(0, 0, 255), -1);
                points_projected++;
            }
        }

        ROS_INFO("Points Projected: %zu", points_projected);

        // 오버레이된 이미지를 sensor_msgs::Image으로 변환
        sensor_msgs::ImagePtr overlay_msg;
        try {
            overlay_msg = cv_bridge::CvImage(img_msg->header, "bgr8", overlay_image).toImageMsg();
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        // 오버레이된 이미지 퍼블리시
        overlay_image_pub_.publish(overlay_msg);
        ROS_DEBUG("Overlay Image published.");
    }

    // 포인트 클라우드를 카메라 프레임으로 변환하는 헬퍼 함수
    bool transformPointCloud(const pcl::PointCloud<PointType>& input_cloud,
                             pcl::PointCloud<PointType>& output_cloud,
                             const std::string& target_frame,
                             const std::string& source_frame,
                             const ros::Time& timestamp)
    {
        try {
            geometry_msgs::TransformStamped transform_stamped = tf_buffer_.lookupTransform(target_frame, source_frame, timestamp, ros::Duration(1.0));
            Eigen::Matrix4d transform_d = Eigen::Matrix4d::Identity();
            transform_d(0, 3) = transform_stamped.transform.translation.x;
            transform_d(1, 3) = transform_stamped.transform.translation.y;
            transform_d(2, 3) = transform_stamped.transform.translation.z;

            Eigen::Quaterniond q(transform_stamped.transform.rotation.w, transform_stamped.transform.rotation.x,
                                 transform_stamped.transform.rotation.y, transform_stamped.transform.rotation.z);
            transform_d.block<3,3>(0,0) = q.toRotationMatrix();
            Eigen::Matrix4f transform_f = transform_d.cast<float>();

            pcl::transformPointCloud(input_cloud, output_cloud, transform_f);

            pcl::VoxelGrid<PointType> voxel_filter;
            voxel_filter.setInputCloud(output_cloud.makeShared());
            voxel_filter.setLeafSize(0.1f, 0.1f, 0.1f);
            pcl::PointCloud<PointType> filtered_cloud;
            voxel_filter.filter(filtered_cloud);

            output_cloud = filtered_cloud;
            ROS_DEBUG("PointCloud transformed and filtered successfully.");
            return true;
        } catch (tf2::TransformException &ex) {
            ROS_WARN("TF transformation failed: %s", ex.what());
            return false;
        }
    }

    // 모든 프레임에 대한 정적 변환을 퍼블리시하는 함수
    void publishStaticTransforms() {
        // world -> ego_car (지면에서 차량 기준점까지의 변환)
        geometry_msgs::TransformStamped world_to_ego;
        world_to_ego.header.stamp = ros::Time::now();
        world_to_ego.header.frame_id = world_frame_;
        world_to_ego.child_frame_id = ego_frame_;
        world_to_ego.transform.translation.x = 0.0;
        world_to_ego.transform.translation.y = 0.0;
        world_to_ego.transform.translation.z = 0.5;  // 차량 기준점 높이 (지면 위)
        world_to_ego.transform.rotation.x = 0.0;
        world_to_ego.transform.rotation.y = 0.0;
        world_to_ego.transform.rotation.z = 0.0;
        world_to_ego.transform.rotation.w = 1.0;

        // ego_car -> hesai_lidar
        geometry_msgs::TransformStamped ego_to_lidar;
        ego_to_lidar.header.stamp = ros::Time::now();
        ego_to_lidar.header.frame_id = ego_frame_;
        ego_to_lidar.child_frame_id = lidar_frame_;
        ego_to_lidar.transform.translation.x = 1.06;  // ego_car 중심에서 전방 1.06m
        ego_to_lidar.transform.translation.y = 0.0;
        ego_to_lidar.transform.translation.z = 1.45;  // 지면으로부터 높이 1.95m - ego_car 높이 0.5m

        double lidar_yaw_rad = -2.1 * M_PI / 180.0;
        Eigen::Quaterniond q_lidar(Eigen::AngleAxisd(lidar_yaw_rad, Eigen::Vector3d::UnitZ()));
        ego_to_lidar.transform.rotation.x = q_lidar.x();
        ego_to_lidar.transform.rotation.y = q_lidar.y();
        ego_to_lidar.transform.rotation.z = q_lidar.z();
        ego_to_lidar.transform.rotation.w = q_lidar.w();

        // ego_car -> gmsl_camera
        geometry_msgs::TransformStamped ego_to_camera;
        ego_to_camera.header.stamp = ros::Time::now();
        ego_to_camera.header.frame_id = ego_frame_;
        ego_to_camera.child_frame_id = camera_frame_;
        ego_to_camera.transform.translation.x = 0.5;  // ego_car 중심에서 전방 0.5m
        ego_to_camera.transform.translation.y = 0.0;
        ego_to_camera.transform.translation.z = 0.7;  // 지면으로부터 높이 1.2m - ego_car 높이 0.5m

        double camera_yaw_rad = -0.3 * M_PI / 180.0;
        Eigen::Quaterniond q_camera(Eigen::AngleAxisd(camera_yaw_rad, Eigen::Vector3d::UnitZ()));
        ego_to_camera.transform.rotation.x = q_camera.x();
        ego_to_camera.transform.rotation.y = q_camera.y();
        ego_to_camera.transform.rotation.z = q_camera.z();
        ego_to_camera.transform.rotation.w = q_camera.w();

        // 모든 정적 변환 퍼블리시
        static_broadcaster_.sendTransform(world_to_ego);
        static_broadcaster_.sendTransform(ego_to_lidar);
        static_broadcaster_.sendTransform(ego_to_camera);

        ROS_INFO("Static transforms published for world to ego_car, ego_car to hesai_lidar, and ego_car to gmsl_camera.");
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "lidar_to_camera_projection");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    // 디버그 로그 레벨 설정
    if (ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug)) {
        ros::console::notifyLoggerLevelsChanged();
    }

    // 노드 초기화
    LidarToCameraProjection projector(nh, pnh);

    // 멀티스레드 스피너 시작
    ros::AsyncSpinner spinner(4);
    spinner.start();

    // 노드 종료 대기
    ros::waitForShutdown();

    return 0;
}
