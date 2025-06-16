import rospy
from visualization_msgs.msg import MarkerArray
import re

# 텍스트 파일에 데이터를 저장하는 함수
def save_to_file(age, v_value):
    with open('marker_data.txt', 'a') as f:
        f.write(f"{age}, {v_value}\n")  # "Age, V" 형식으로 저장

def callback(msg):
    # 'markers'에서 'text' 필드를 파싱하여 'Age'와 'V' 값을 추출
    for marker in msg.markers:
        marker_text = marker.text

        # 'Age' 값 추출
        age_match = re.search(r'Age:\s*(\d+)', marker_text)
        if age_match:
            age = int(age_match.group(1))
        
        # 'V' 값 추출
        v_match = re.search(r'V:\s*(-?\d+)', marker_text)
        if v_match:
            v_value = int(v_match.group(1))

            # Age와 V 값을 파일에 저장
            save_to_file(age, v_value)

def listener():
    rospy.init_node('v_value_listener', anonymous=True)
    rospy.Subscriber("/mobinha/visualize/visualize/track_text", MarkerArray, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
