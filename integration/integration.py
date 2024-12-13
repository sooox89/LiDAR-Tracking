#!/usr/bin/env python3

import os
import csv
from scipy.interpolate import interp1d
from scipy.spatial import KDTree
import numpy as np
import pymap3d
import math
import tf
import tf2_ros
import rospy
import roslib
from std_msgs.msg import Header, ColorRGBA
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from novatel_oem7_msgs.msg import INSPVA
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, TransformStamped, Pose, Vector3, Quaternion
from jsk_rviz_plugins.msg import OverlayText
from scipy.spatial.transform import Rotation as R

from utils import *
# lidar_tracking 패키지의 경로를 가져옴
package_path = roslib.packages.get_pkg_dir('lidar_tracking')
dae_path = os.path.join(package_path, 'urdf/car.dae')  # car.dae 파일 경로 설정
map_path = os.path.join(package_path, 'map/songdo.json') # 지도 파일 경로 설정



# ioniq calibration 
t_gps_lidar = np.array([1.06, 0, 1.22]) # GPS와 LiDAR 간의 변화 벡터 
q_gps_lidar = rotate_quaternion_yaw((0, 0, 0, 1), -2.1) # GPS와 LiDAR 간의 회전 쿼터니언
t_gps_ego = np.array([1.5275, 0, 0]) # GPS와 ego(자차) 간의 변환 벡터
q_gps_ego = rotate_quaternion_yaw((0, 0, 0, 1), -0.3) # GPS와 ego(자차) 간의 회전 쿼터니언
t_lidar_camera = np.array([1.4, 0.1, -0.3])  # 오프셋 조정 예시
# 쿼터니언 설정

def rotate_quaternion_for_lidar_camera():
    # 축 맞춤을 위한 회전 쿼터니언 계산
    rotation_matrix = np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ])
    r = R.from_matrix(rotation_matrix)
    q = r.as_quat()  # [x, y, z, w] 형식의 쿼터니언 반환
    return (q[0], q[1], q[2], q[3])

# 쿼터니언 설정
q_lidar_camera = rotate_quaternion_for_lidar_camera()

class Integration:
    def __init__(self):
        rospy.init_node('Integration')
        
        # lanelet 지도 로드 및 시각화
        self.interp_distance = 2.0
        self.lmap = LaneletMap(map_path, self.interp_distance)
        lanelet_map_viz = LaneletMapViz(self.lmap.lanelets, self.lmap.for_viz)
        pub_lanelet_map = rospy.Publisher('/lanelet_map', MarkerArray, queue_size=1, latch=True)
        pub_lanelet_map.publish(lanelet_map_viz)
        
        # # microlanelet
        # self.mlgraph = MicroLaneletGraph(self.lmap, 15.0)
        # micro_lanelet_graph_viz = MicroLaneletGraphViz(self.lmap.lanelets, self.mlgraph.graph)
        # pub_micro_lanelet_graph = rospy.Publisher('/micro_lanelet_graph', MarkerArray, queue_size=1, latch=True)
        # pub_micro_lanelet_graph.publish(micro_lanelet_graph_viz)

        # waypoints
        self.use_waypoints = True
        self.r = 100.0 # 로컬 웨이 포인트를 검색할 반경
        self.build_waypoint_kdtree() # KD-Tree 구축
        self.pub_waypoints = rospy.Publisher('/waypoints', PointCloud2, queue_size=1)

        # ego car marker
        self.ego_car = self.egoCar()
        self.pub_ego_car = rospy.Publisher('/car_model', Marker, queue_size=1)
        
        # calibration (센서 간의 정적 변환 설정)
        self.br = tf.TransformBroadcaster()     # 실시간 변환을 위한 브로드캐스터
        self.static_br = tf2_ros.StaticTransformBroadcaster()   # 정적 변환을 위한 브로드 캐스터 
        static_transforms = [
            # ioniq 차량의 센서 간 변환 설정
            (t_gps_ego, q_gps_ego, 'ego_car', 'gps'),
            (t_gps_lidar, q_gps_lidar, 'hesai_lidar', 'gps'),
            (t_lidar_camera, q_lidar_camera, 'camera', 'hesai_lidar')
        ]
        self.publish_static_tfs(static_transforms) # 정적 변환 전송

        # ego information
        self.pub_ego_info = rospy.Publisher('/car_info', OverlayText, queue_size=1)

        # evaluation
        self.save_flag = True
        self.file_name = 'ioniq.csv'
        self.csv_file = open(self.file_name, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['rostime', 'gpstime', 'world_x', 'world_y', 'azimuth', 'vx', 'vy'])
        rospy.on_shutdown(self.shutdown_hook)  # 노드 종료 시 파일 닫기

        rospy.Subscriber('/novatel/oem7/inspva', INSPVA, self.novatel_cb)

        rospy.loginfo("Initialized")

    def shutdown_hook(self):
        self.csv_file.close()

    def egoCar(self):
        marker = Marker(
            header=Header(frame_id='ego_car'),
            ns='ego_car',
            id=0,
            type=Marker.MESH_RESOURCE,
            mesh_resource="file://" + dae_path,
            action=Marker.ADD,
            lifetime=rospy.Duration(0.05),
            scale=Vector3(x=2.0, y=2.0, z=2.0),
            color=ColorRGBA(r=0.7, g=0.7, b=0.7, a=1.0),
            pose=Pose(
                position=Point(x=0, y=0, z=1.0),
                orientation=Quaternion(*tf.transformations.quaternion_from_euler(0, 0, math.radians(90)))
            )
        )
        return marker

    def egoInfo(self, x, y, azimuth, vx, vy):
        # 자차 정보 텍스트 설정
        # text = "Position:\nx: {:.2f}\ny: {:.2f}\nazimuth: {:.2f}\n\nSpeed:\nvx: {:.2f} m/s\nvy: {:.2f} m/s".format(x, y, azimuth, vx, vy)
        v = math.sqrt(vx**2 + vy**2) # 속도 크기 계산
        v = v * 3.6 # 속도 m/s에서 km/h로 단위 변경
        text = "Position:\nx: {:.2f}\ny: {:.2f}\nazimuth: {:.2f}\n\nSpeed: {:.2f} km/h".format(x, y, azimuth, v)
        overlay_text = OverlayText()
        overlay_text.action = OverlayText.ADD
        overlay_text.width = 400
        overlay_text.height = 200
        overlay_text.left = 10  # 왼쪽에서부터의 위치
        overlay_text.top = 10   # 위쪽에서부터의 위치
        overlay_text.text_size = 14
        overlay_text.line_width = 2
        overlay_text.font = "DejaVu Sans Mono"
        overlay_text.text = text
        overlay_text.fg_color = ColorRGBA(0.0, 1.0, 0.0, 1.0)  # 녹색 글자
        overlay_text.bg_color = ColorRGBA(0.0, 0.0, 0.0, 0.5)  # 반투명 검정 배경
        return overlay_text

    def publish_static_tfs(self, transforms):
        # 정적 변환 전송
        static_transformStamped_vec = []
        for translation, rotation, child_frame, parent_frame in transforms:
            static_transformStamped = TransformStamped()
            static_transformStamped.header.frame_id = parent_frame
            static_transformStamped.child_frame_id = child_frame
            static_transformStamped.transform.translation.x = translation[0]
            static_transformStamped.transform.translation.y = translation[1]
            static_transformStamped.transform.translation.z = translation[2]
            static_transformStamped.transform.rotation.x = rotation[0]
            static_transformStamped.transform.rotation.y = rotation[1]
            static_transformStamped.transform.rotation.z = rotation[2]
            static_transformStamped.transform.rotation.w = rotation[3]
            static_transformStamped_vec.append(static_transformStamped)
        
        self.static_br.sendTransform(static_transformStamped_vec) # 변환 브로드캐스트 

    def novatel_cb(self, msg):
        # INSPVA 메시지 콜백 함수 
        self.timestamp = msg.header.stamp
        self.latitude = msg.latitude
        self.longitude = msg.longitude
        self.altitude = msg.height
        self.x, self.y, self.z = pymap3d.geodetic2enu(
            msg.latitude, msg.longitude, 0, self.lmap.base_lla[0], self.lmap.base_lla[1], 0)
        self.roll = msg.roll
        self.pitch = msg.pitch
        self.azimuth = (90 - msg.azimuth) % 360 # 방위각 보정
        
        # 쿼터니언 계산 (roll, pitch, yaw)
        quaternion = tf.transformations.quaternion_from_euler(
            math.radians(self.roll), math.radians(self.pitch), math.radians(self.azimuth))  # RPY
        # world 프레임과 gps 프레임 간의 변환 브로드캐스트 
        self.br.sendTransform(
            (self.x, self.y, self.z),
            (quaternion[0], quaternion[1],
                quaternion[2], quaternion[3]),
            self.timestamp,
            'gps',
            'world'
        )

        self.pub_ego_car.publish(self.ego_car) # 자차 모델 마커 퍼블러시 
        
        # 차량 좌표계로 변환하기 위한 회전 행렬의 요소 계산
        azimuth_rad_original = math.radians(msg.azimuth)
        cos_azimuth = math.cos(azimuth_rad_original)
        sin_azimuth = math.sin(azimuth_rad_original)

        # 차량 좌표계에서의 속도 성분 계산
        vx = msg.north_velocity * cos_azimuth + msg.east_velocity * sin_azimuth
        vy = -msg.north_velocity * sin_azimuth + msg.east_velocity * cos_azimuth

        self.ego_info = self.egoInfo(self.x, self.y, self.azimuth, vx, vy)

        # evaluation 
        if self.save_flag == False:
            gps_time = gpsTime(msg.nov_header.gps_week_number, msg.nov_header.gps_week_milliseconds)
            t_world_gps = [self.x, self.y, self.z]
            q_world_gps = quaternion
            R_world_gps = tf.transformations.quaternion_matrix(q_world_gps)[:3, :3]
            t_gps_lidar_in_world = R_world_gps.dot(t_gps_lidar)
            t_world_lidar = t_world_gps + t_gps_lidar_in_world
            q_world_lidar = tf.transformations.quaternion_multiply(q_world_gps, q_gps_lidar)
            _, _, yaw_lidar = tf.transformations.euler_from_quaternion(q_world_lidar)
            azimuth_lidar = (math.degrees(yaw_lidar) + 360) % 360

            if self.file_name == "ioniq.csv": # ioniq 
                self.csv_writer.writerow([self.timestamp.to_sec(), gps_time, t_world_lidar[0], t_world_lidar[1], azimuth_lidar, vx, vy])
            elif self.file_name == "avente.csv": # avente
                self.csv_writer.writerow([self.timestamp.to_sec(), gps_time, self.x, self.y, self.azimuth, vx, vy])

            self.ego_info = self.egoInfo(t_world_lidar[0], t_world_lidar[1], azimuth_lidar, vx, vy)

        self.pub_ego_info.publish(self.ego_info) # 자차 정보 텍스트 퍼블리시 
        ####################################################
        self.update_local_waypoints(self.r) # 로컬 웨이 포인트 업데이트 

    def build_waypoint_kdtree(self): 
        # 모든 웨이포인트를 수집하여 KD-Tree를 구축
        all_waypoints = []
        for id_, lanelet in self.lmap.lanelets.items():
            waypoints = lanelet['interp_waypoints']
            all_waypoints.extend(waypoints)
        self.waypoints_np = np.array(all_waypoints)
        self.kdtree = KDTree(self.waypoints_np)

    def update_local_waypoints(self, r):
        # 로컬 웨이포인트를 업데이트하여 퍼블리시
        if self.use_waypoints == False:
            return

        if not hasattr(self, 'kdtree'):
            rospy.logerr("KD-Tree is not built yet.")
            return
        # 현재 위치를 중심으로 반경 r 내의 웨이포인트 인덱스 검색
        indices = self.kdtree.query_ball_point([self.x, self.y], r)

        if not indices:
            return

        nearby_waypoints = self.waypoints_np[indices]
        
        # 자차의 방위각을 라디안으로 변환
        azimuth_vehicle = self.azimuth
        azimuth_rad = math.radians(azimuth_vehicle)
        cos_azimuth = math.cos(-azimuth_rad)
        sin_azimuth = math.sin(-azimuth_rad)
        
        # 웨이포인트를 자차 좌표계로 변환
        dx = nearby_waypoints[:, 0] - self.x
        dy = nearby_waypoints[:, 1] - self.y

        x_e = dx * cos_azimuth - dy * sin_azimuth
        y_e = dx * sin_azimuth + dy * cos_azimuth

        transformed_waypoints = list(zip(x_e, y_e, np.zeros_like(x_e)))
        
        # PointCloud2 메시지 생성
        point_cloud = pc2.create_cloud_xyz32(
            header=rospy.Header(frame_id='ego_car', stamp=self.timestamp),
            points=transformed_waypoints
        )

        self.pub_waypoints.publish(point_cloud)  # 웨이포인트 퍼블리시

def main():
    integration = Integration()  # Integration 클래스 인스턴스 생성
    rospy.spin() # ROS 노드 실행

if __name__ == '__main__':
    main()
