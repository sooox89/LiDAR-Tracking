#!/usr/bin/env python3

import rospy
from novatel_oem7_msgs.msg import INSPVA
from sensor_msgs.msg import Imu
import tf

def inspva_callback(data):
    imu_msg = Imu()
    imu_msg.header.stamp = rospy.Time.now()
    imu_msg.header.frame_id = "imu_link"  # 필요에 따라 변경

    # Roll, Pitch, Azimuth(Yaw)를 사용해 쿼터니언 계산
    quaternion = tf.transformations.quaternion_from_euler(data.roll, data.pitch, data.azimuth)
    imu_msg.orientation.x = quaternion[0]
    imu_msg.orientation.y = quaternion[1]
    imu_msg.orientation.z = quaternion[2]
    imu_msg.orientation.w = quaternion[3]

    # 각속도 및 선가속도 사용 불가 (-1로 설정)
    imu_msg.angular_velocity_covariance[0] = -1
    imu_msg.linear_acceleration_covariance[0] = -1

    imu_pub.publish(imu_msg)

if __name__ == "__main__":
    rospy.init_node('inspva_to_imu')
    imu_pub = rospy.Publisher('/imu/data_raw', Imu, queue_size=10)
    rospy.Subscriber('/novatel/oem7/inspva', INSPVA, inspva_callback)
    rospy.spin()
