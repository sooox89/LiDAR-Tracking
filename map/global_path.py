import json
import rospy
from visualization_msgs.msg import Marker

def marker_callback(msg):
    transformed_data = {}

    # Marker 메시지의 각 요소를 JSON 형식에 맞게 변환
    for index, point in enumerate(msg.points):
        transformed_data[str(index)] = [
            point.x,    # X 좌표
            point.y,    # Y 좌표
            "driving",  # 활동
            5           # 정밀도
        ]
    
    # JSON 파일로 저장
    with open('global_path_data.json', 'w', encoding='utf-8') as file:
        json.dump(transformed_data, file, indent=4)
    
    rospy.loginfo("global_path_data.json 파일에 데이터가 저장되었습니다.")

def main():
    rospy.init_node('global_path_to_json', anonymous=True)
    
    # Subscriber 생성
    rospy.Subscriber('/mobinha/global_path', Marker, marker_callback)
    
    # 노드가 종료될 때까지 대기
    rospy.spin()

if __name__ == '__main__':
    main()
