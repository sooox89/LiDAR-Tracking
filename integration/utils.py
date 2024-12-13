#!/usr/bin/env python
import os
import copy
from scipy.interpolate import interp1d
from scipy.spatial import KDTree
import numpy as np
import json
import math
import tf
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

# 평가 관련 함수 및 클래스

def gpsTime(gps_week_number, gps_week_milliseconds):
    """
    GPS 주 번호와 주 내 밀리초를 UNIX 타임스탬프로 변환하는 함수.
    
    :param gps_week_number: GPS 주 번호 (정수)
    :param gps_week_milliseconds: GPS 주 내 밀리초 (정수)
    :return: 변환된 UNIX 시간 (float)
    """
    gps_epoch_unix = 315964800  # GPS 에포크 시간 (1980-01-06)을 UNIX 타임스탬프로
    gps_seconds = gps_week_number * 604800 + gps_week_milliseconds / 1000.0  # 주 번호와 밀리초를 초로 변환
    gps_time = gps_epoch_unix + gps_seconds  # GPS 시간을 UNIX 시간으로 변환
    return gps_time

def rotate_quaternion_yaw(quaternion, yaw_degrees):
    """
    주어진 쿼터니언에 Yaw(회전 각도)를 추가하여 새로운 쿼터니언을 생성하는 함수.
    
    :param quaternion: 기존 쿼터니언 [x, y, z, w] (리스트 또는 튜플)
    :param yaw_degrees: 추가할 Yaw 각도 (도 단위)
    :return: 회전된 새로운 쿼터니언 [x, y, z, w] (리스트)
    """
    yaw_radians = math.radians(yaw_degrees)  # 도 단위를 라디안으로 변환
    q_yaw = tf.transformations.quaternion_from_euler(0, 0, yaw_radians)  # Yaw만 적용된 쿼터니언 생성
    return tf.transformations.quaternion_multiply(quaternion, q_yaw)  # 기존 쿼터니언과 곱하여 새로운 쿼터니언 생성

def Bound(ns, id_, n, points, type_, color):
    """
    경계선을 시각화하기 위한 마커 생성 함수.
    
    :param ns: 네임스페이스 (문자열)
    :param id_: 식별자 (정수 또는 문자열)
    :param n: 인덱스 (정수)
    :param points: 경계선의 점들 (리스트 of 튜플)
    :param type_: 마커 유형 ('solid' 또는 'dotted')
    :param color: 색상 (리스트 [R, G, B, A])
    :return: 생성된 마커 객체
    """
    if type_ == 'solid':
        marker = Line('%s_%s' % (ns, id_), n, 0.15, color)  # 실선 마커 생성
        for pt in points:
            marker.points.append(Point(x=pt[0], y=pt[1], z=0.0))  # 점 추가

    elif type_ == 'dotted':
        marker = Points('%s_%s' % (ns, id_), n, 0.15, color)  # 점 마커 생성
        for pt in points:
            marker.points.append(Point(x=pt[0], y=pt[1], z=0.0))  # 점 추가

    return marker

def Points(ns, id_, scale, color):
    """
    POINTS 유형의 마커를 생성하는 함수.
    
    :param ns: 네임스페이스 (문자열)
    :param id_: 식별자 (정수)
    :param scale: 스케일 (float)
    :param color: 색상 (리스트 [R, G, B, A])
    :return: 생성된 Marker 객체
    """
    marker = Marker()
    marker.type = Marker.POINTS  # 마커 유형 설정
    marker.action = Marker.ADD  # 마커 액션 설정
    marker.header.frame_id = 'world'  # 프레임 ID 설정
    marker.ns = ns  # 네임스페이스 설정
    marker.id = id_  # 식별자 설정
    marker.lifetime = rospy.Duration(0)  # 지속 시간 설정 (0은 영구)
    marker.scale.x = scale  # 스케일 설정
    marker.scale.y = scale
    marker.color.r = color[0]  # 색상 설정
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3]
    return marker

def Line(ns, id_, scale, color):
    """
    LINE_STRIP 유형의 마커를 생성하는 함수.
    
    :param ns: 네임스페이스 (문자열)
    :param id_: 식별자 (정수)
    :param scale: 스케일 (float)
    :param color: 색상 (리스트 [R, G, B, A])
    :return: 생성된 Marker 객체
    """
    marker = Marker()
    marker.type = Marker.LINE_STRIP  # 마커 유형 설정
    marker.action = Marker.ADD  # 마커 액션 설정
    marker.header.frame_id = 'world'  # 프레임 ID 설정
    marker.ns = ns  # 네임스페이스 설정
    marker.id = id_  # 식별자 설정
    marker.lifetime = rospy.Duration(0)  # 지속 시간 설정 (0은 영구)
    marker.scale.x = scale  # 선의 두께 설정
    marker.color.r = color[0]  # 색상 설정
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3]
    marker.pose.orientation.x = 0.0  # 포즈 방향 설정
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0
    return marker

def Sphere(ns, id_, data, scale, color):
    """
    SPHERE 유형의 마커를 생성하는 함수.
    
    :param ns: 네임스페이스 (문자열)
    :param id_: 식별자 (정수)
    :param data: 위치 데이터 [x, y]
    :param scale: 스케일 (float)
    :param color: 색상 (리스트 [R, G, B, A])
    :return: 생성된 Marker 객체
    """
    marker = Marker()
    marker.type = Marker.SPHERE  # 마커 유형 설정
    marker.action = Marker.ADD  # 마커 액션 설정
    marker.header.frame_id = 'world'  # 프레임 ID 설정
    marker.ns = ns  # 네임스페이스 설정
    marker.id = id_  # 식별자 설정
    marker.lifetime = rospy.Duration(0)  # 지속 시간 설정 (0은 영구)
    marker.scale.x = scale  # 스케일 설정
    marker.scale.y = scale
    marker.scale.z = scale
    marker.color.r = color[0]  # 색상 설정
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3]
    marker.pose.position.x = data[0]  # 위치 설정
    marker.pose.position.y = data[1]
    marker.pose.position.z = 1.0  # 높이 설정 (고정)
    return marker

def Node(id_, n, pt, color):
    """
    텍스트 마커를 생성하여 노드를 시각화하는 함수.
    
    :param id_: 노드 ID (문자열)
    :param n: 인덱스 (정수)
    :param pt: 위치 [x, y]
    :param color: 색상 (리스트 [R, G, B, A])
    :return: 생성된 Marker 객체
    """
    marker = Text('graph_id', n, 2.5, color, id_)  # 텍스트 마커 생성
    marker.pose.position = Point(x=pt[0], y=pt[1], z=1.0)  # 위치 설정
    return marker

def Edge(n, points, color):
    """
    엣지(경로)를 시각화하기 위한 마커 생성 함수.
    
    :param n: 인덱스 (정수)
    :param points: 점들 (리스트 of 튜플)
    :param color: 색상 (리스트 [R, G, B, A])
    :return: 생성된 두 개의 Marker 객체 (라인과 화살표)
    """
    if len(points) == 2:
        wx, wy = zip(*points)
        itp = QuadraticSplineInterpolate(list(wx), list(wy))  # 쿼드라틱 스플라인 보간
        pts = []
        for ds in np.arange(0.0, itp.s[-1], 0.5):
            pts.append(itp.calc_position(ds))  # 보간된 점들 추가
        points = pts

    marker1 = Line('edge_line', n, 0.5, color)  # 라인 마커 생성
    for pt in points:
        marker1.points.append(Point(x=pt[0], y=pt[1], z=0.0))  # 점 추가

    marker2 = Arrow('edge_arrow', n, (1.0, 2.0, 4.0), color)  # 화살표 마커 생성
    num = len(points)
    if num > 2:
        marker2.points.append(
            Point(x=points[-min(max(num, 3), 5)][0], y=points[-min(max(num, 3), 5)][1]))  # 화살표 시작점
    else:
        marker2.points.append(Point(x=points[-2][0], y=points[-2][1]))
    marker2.points.append(Point(x=points[-1][0], y=points[-1][1]))  # 화살표 끝점
    return marker1, marker2

def Text(ns, id_, scale, color, text):
    """
    TEXT_VIEW_FACING 유형의 마커를 생성하는 함수.
    
    :param ns: 네임스페이스 (문자열)
    :param id_: 식별자 (정수)
    :param scale: 스케일 (float)
    :param color: 색상 (리스트 [R, G, B, A])
    :param text: 표시할 텍스트 (문자열)
    :return: 생성된 Marker 객체
    """
    marker = Marker()
    marker.type = Marker.TEXT_VIEW_FACING  # 마커 유형 설정
    marker.action = Marker.ADD  # 마커 액션 설정
    marker.header.frame_id = 'world'  # 프레임 ID 설정
    marker.ns = ns  # 네임스페이스 설정
    marker.id = id_  # 식별자 설정
    marker.lifetime = rospy.Duration(0)  # 지속 시간 설정 (0은 영구)
    marker.text = text  # 텍스트 설정
    marker.scale.z = scale  # 텍스트 크기 설정
    marker.color.r = color[0]  # 색상 설정
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3]
    marker.pose.orientation.x = 0.0  # 포즈 방향 설정
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0
    return marker

def Arrow(ns, id_, scale, color):
    """
    ARROW 유형의 마커를 생성하는 함수.
    
    :param ns: 네임스페이스 (문자열)
    :param id_: 식별자 (정수)
    :param scale: 스케일 (튜플 (float, float, float))
    :param color: 색상 (리스트 [R, G, B, A])
    :return: 생성된 Marker 객체
    """
    marker = Marker()
    marker.type = Marker.ARROW  # 마커 유형 설정
    marker.action = Marker.ADD  # 마커 액션 설정
    marker.header.frame_id = 'world'  # 프레임 ID 설정
    marker.ns = ns  # 네임스페이스 설정
    marker.id = id_  # 식별자 설정
    marker.lifetime = rospy.Duration(0)  # 지속 시간 설정 (0은 영구)
    marker.scale.x = scale[0]  # 화살표 길이 설정
    marker.scale.y = scale[1]  # 화살표 너비 설정
    marker.scale.z = scale[2]  # 화살표 높이 설정
    marker.color.r = color[0]  # 색상 설정
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3]
    marker.pose.orientation.x = 0.0  # 포즈 방향 설정
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0
    return marker

def LaneletMapViz(lanelet, for_viz):
    """
    레인렛 맵을 시각화하기 위한 마커 배열을 생성하는 함수.
    
    :param lanelet: 레인렛 데이터 (딕셔너리)
    :param for_viz: 시각화를 위한 추가 데이터 (리스트)
    :return: 생성된 MarkerArray 객체
    """
    array = MarkerArray()
    for id_, data in lanelet.items():
        # 왼쪽 경계선 시각화
        for n, (leftBound, leftType) in enumerate(zip(data['leftBound'], data['leftType'])):
            marker = Bound('leftBound', id_, n, leftBound,
                           leftType, (1.0, 1.0, 1.0, 1.0))
            array.markers.append(marker)

        # 오른쪽 경계선 시각화
        for n, (rightBound, rightType) in enumerate(zip(data['rightBound'], data['rightType'])):
            marker = Bound('rightBound', id_, n, rightBound,
                           rightType, (1.0, 1.0, 1.0, 1.0))
            array.markers.append(marker)

    # 추가 시각화 요소 (예: 정지선 등)
    for n, (points, type_) in enumerate(for_viz):
        if type_ == 'stop_line':
            marker = Bound('for_viz', n, n, points,
                           'solid', (1.0, 1.0, 1.0, 1.0))
            array.markers.append(marker)
        else:
            marker = Bound('for_viz', n, n, points,
                           type_, (1.0, 1.0, 1.0, 1.0))
            array.markers.append(marker)

    return array

def MicroLaneletGraphViz(lanelet, graph):
    """
    마이크로 레인렛 그래프를 시각화하기 위한 마커 배열을 생성하는 함수.
    
    :param lanelet: 레인렛 데이터 (딕셔너리)
    :param graph: 그래프 데이터 (딕셔너리)
    :return: 생성된 MarkerArray 객체
    """
    array = MarkerArray()

    for n, (node_id, data) in enumerate(graph.items()):
        split = node_id.split('_')

        if len(split) == 1:
            id_ = split[0]
            from_idx = lanelet[id_]['idx_num'] // 2  # 시작 인덱스 계산
            from_pts = lanelet[id_]['waypoints']
            marker = Node(node_id, n, from_pts[from_idx], (1.0, 1.0, 1.0, 1.0))  # 노드 마커 생성
            array.markers.append(marker)

            for m, target_node_id in enumerate(data.keys()):
                split = target_node_id.split('_')
                pts = []

                if len(split) == 1:
                    target_id = split[0]
                    to_pts = lanelet[target_id]['waypoints']
                    to_idx = lanelet[target_id]['idx_num'] // 2
                    pts.extend(from_pts[from_idx:])
                    pts.extend(to_pts[:to_idx])
                else:
                    target_id = split[0]
                    cut_n = int(split[1])
                    to_pts = lanelet[target_id]['waypoints']
                    to_idx = sum(lanelet[target_id]['cut_idx'][cut_n]) // 2
                    pts.extend(from_pts[from_idx:])
                    pts.extend(to_pts[:to_idx])

                marker1, marker2 = Edge(n*100000+m, pts, (0.0, 1.0, 0.0, 0.5))  # 엣지 마커 생성
                array.markers.append(marker1)
                array.markers.append(marker2)

        else:
            id_ = split[0]
            cut_n = int(split[1])
            from_idx = sum(lanelet[id_]['cut_idx'][cut_n]) // 2  # 시작 인덱스 계산
            from_pts = lanelet[id_]['waypoints']
            marker = Node(node_id, n, from_pts[from_idx], (1.0, 1.0, 1.0, 1.0))  # 노드 마커 생성
            array.markers.append(marker)

            for m, target_node_id in enumerate(data.keys()):
                split = target_node_id.split('_')
                pts = []

                if len(split) == 1:
                    target_id = split[0]
                    to_pts = lanelet[target_id]['waypoints']
                    to_idx = lanelet[target_id]['idx_num'] // 2
                    pts.extend(from_pts[from_idx:])
                    pts.extend(to_pts[:to_idx])
                else:
                    target_id = split[0]
                    cut_n = int(split[1])
                    to_pts = lanelet[target_id]['waypoints']
                    to_idx = sum(lanelet[target_id]['cut_idx'][cut_n]) // 2
                    pts = [from_pts[from_idx], to_pts[to_idx]]

                marker1, marker2 = Edge(n*100000+m, pts, (0.0, 1.0, 0.0, 0.5))  # 엣지 마커 생성
                array.markers.append(marker1)
                array.markers.append(marker2)

    return array

def euc_distance(pt1, pt2):
    """
    두 점 간의 유클리드 거리 계산 함수.
    
    :param pt1: 첫 번째 점 [x, y]
    :param pt2: 두 번째 점 [x, y]
    :return: 유클리드 거리 (float)
    """
    return np.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2)

def find_nearest_idx(pts, pt):
    """
    주어진 점(pt)과 가장 가까운 점의 인덱스를 찾는 함수.
    
    :param pts: 점들의 리스트 (리스트 of 튜플)
    :param pt: 기준 점 [x, y]
    :return: 가장 가까운 점의 인덱스 (정수)
    """
    min_dist = float('inf')  # 최소 거리 초기화
    min_idx = 0

    for idx, pt1 in enumerate(pts):
        dist = euc_distance(pt1, pt)  # 거리 계산
        if dist < min_dist:
            min_dist = dist
            min_idx = idx

    return min_idx

class QuadraticSplineInterpolate:
    """
    2차 스플라인 보간을 위한 클래스.
    """
    def __init__(self, x, y):
        self.s = self.calc_s(x, y)  # 축적 거리 계산
        self.sx = interp1d(self.s, x, fill_value="extrapolate")  # x에 대한 보간 함수
        self.sy = interp1d(self.s, y, fill_value="extrapolate")  # y에 대한 보간 함수

    def calc_s(self, x, y):
        """
        축적 거리를 계산하는 함수.
        
        :param x: x 좌표 리스트
        :param y: y 좌표 리스트
        :return: 축적 거리 리스트
        """
        dx = np.diff(x)  # x 차분
        dy = np.diff(y)  # y 차분
        self.ds = [math.sqrt(idx ** 2 + idy ** 2) for (idx, idy) in zip(dx, dy)]  # 각 구간의 거리 계산
        s = [0]  # 시작 거리
        s.extend(np.cumsum(self.ds))  # 누적 거리 계산
        return s

    def calc_d(self, sp, x):
        """
        1차 도함수를 계산하는 함수.
        
        :param sp: 보간 함수
        :param x: 축적 거리
        :return: 도함수 값
        """
        dx = 1.0
        dp = sp(x + dx)  # x+dx에서의 함수 값
        dm = sp(x - dx)  # x-dx에서의 함수 값
        d = (dp - dm) / dx  # 중앙 차분 방식으로 도함수 계산
        return d

    def calc_dd(self, sp, x):
        """
        2차 도함수를 계산하는 함수.
        
        :param sp: 보간 함수
        :param x: 축적 거리
        :return: 2차 도함수 값
        """
        dx = 2.0
        ddp = self.calc_d(sp, x + dx)  # x+dx에서의 도함수 값
        ddm = self.calc_d(sp, x - dx)  # x-dx에서의 도함수 값
        dd = (ddp - ddm) / dx  # 중앙 차분 방식으로 2차 도함수 계산
        return dd

    def calc_yaw(self, s):
        """
        주어진 축적 거리에서의 Yaw 각도를 계산하는 함수.
        
        :param s: 축적 거리
        :return: Yaw 각도 (라디안)
        """
        dx = self.calc_d(self.sx, s)  # x에 대한 도함수
        dy = self.calc_d(self.sy, s)  # y에 대한 도함수
        yaw = math.atan2(dy, dx)  # Yaw 각도 계산
        return yaw

    def calc_position(self, s):
        """
        주어진 축적 거리에서의 위치를 계산하는 함수.
        
        :param s: 축적 거리
        :return: 위치 (x, y) 튜플
        """
        x = self.sx(s)  # x 좌표 보간
        y = self.sy(s)  # y 좌표 보간
        return x, y

    def calc_curvature(self, s):
        """
        주어진 축적 거리에서의 곡률을 계산하는 함수.
        
        :param s: 축적 거리
        :return: 곡률 (float)
        """
        dx = self.calc_d(self.sx, s)  # x에 대한 1차 도함수
        ddx = self.calc_dd(self.sx, s)  # x에 대한 2차 도함수
        dy = self.calc_d(self.sy, s)  # y에 대한 1차 도함수
        ddy = self.calc_dd(self.sy, s)  # y에 대한 2차 도함수
        k = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2)**(3 / 2))  # 곡률 계산
        return k

class LaneletMap:
    def __init__(self, map_path, interp_distance=0.5):
        try:
            with open(map_path, 'r') as f:
                map_data = json.load(f)
        except FileNotFoundError:
            rospy.logerr(f"File not found: {map_path}")
            raise
        except json.JSONDecodeError:
            rospy.logerr(f"Invalid JSON format in file: {map_path}")
            raise

        # JSON 데이터 검증
        required_keys = ['lanelets', 'groups', 'precision', 'for_vis', 'base_lla']
        for key in required_keys:
            if key not in map_data:
                rospy.logerr(f"Missing key in JSON: {key}")
                raise KeyError(f"Missing key: {key}")

        self.map_data = map_data
        self.lanelets = map_data['lanelets']
        self.groups = map_data['groups']
        self.precision = map_data['precision']
        self.for_viz = map_data['for_vis']
        self.base_lla = map_data['base_lla']

        self.interp_distance = interp_distance
        self.preprocess_lanelets()
        
    def preprocess_lanelets(self):
        """
        레인렛 데이터의 웨이포인트를 보간하여 정규화된 웨이포인트를 생성.
        """
        for id_, lanelet in self.lanelets.items():
            waypoints = np.array(lanelet['waypoints'])  # 웨이포인트 배열
            if len(waypoints) < 2:
                continue  # 웨이포인트가 2개 미만인 경우 스킵
            wp_x = waypoints[:, 0]  # x 좌표 추출
            wp_y = waypoints[:, 1]  # y 좌표 추출

            delta_wp = np.diff(waypoints, axis=0)  # 웨이포인트 간 차분
            segment_lengths = np.hypot(delta_wp[:, 0], delta_wp[:, 1])  # 각 세그먼트의 길이 계산
            distances_along_wp = np.concatenate(([0], np.cumsum(segment_lengths)))  # 누적 거리 계산

            total_length = distances_along_wp[-1]  # 총 길이
            num_points = int(total_length / self.interp_distance) + 1  # 보간할 점의 개수
            if num_points < 2:
                num_points = 2  # 최소 2개의 포인트 유지
            s_interp = np.linspace(0, total_length, num_points)  # 보간할 축적 거리 생성

            x_interp = np.interp(s_interp, distances_along_wp, wp_x)  # x 보간
            y_interp = np.interp(s_interp, distances_along_wp, wp_y)  # y 보간

            lanelet['interp_waypoints'] = np.stack((x_interp, y_interp), axis=-1)  # 보간된 웨이포인트 저장

class MicroLaneletGraph:
    """
    마이크로 레인렛 그래프를 생성하고 관리하는 클래스.
    """
    def __init__(self, lmap, cut_dist):
        """
        마이크로 레인렛 그래프 초기화.
        
        :param lmap: LaneletMap 객체
        :param cut_dist: 절단 거리 (float)
        """
        self.cut_dist = cut_dist  # 절단 거리 설정
        self.precision = lmap.precision  # 정밀도
        self.lanelets = lmap.lanelets  # 레인렛 데이터
        self.groups = lmap.groups  # 그룹 데이터
        self.generate_micro_lanelet_graph()  # 마이크로 레인렛 그래프 생성

    def generate_micro_lanelet_graph(self):
        """
        마이크로 레인렛 그래프를 생성하는 함수.
        """
        self.graph = {}
        cut_idx = int(self.cut_dist / self.precision)  # 절단 인덱스 계산

        for group in self.groups:
            group = copy.copy(group)  # 그룹 데이터 복사

            # 레인렛의 길이를 기준으로 그룹을 정렬
            if self.lanelets[group[0]]['length'] > self.lanelets[group[-1]]['length']:
                group.reverse()

            idx_num = self.lanelets[group[0]]['idx_num']  # 인덱스 개수

            cut_num = idx_num // cut_idx  # 절단 횟수 계산
            if idx_num % cut_idx != 0:
                cut_num += 1  # 나머지가 있으면 절단 횟수 증가

            for n, id_ in enumerate(group):
                self.lanelets[id_]['cut_idx'] = []  # 절단 인덱스 초기화

                if n == 0:
                    # 첫 번째 레인렛의 절단 인덱스 설정
                    for i in range(cut_num):
                        start_idx = i * cut_idx

                        if i == cut_num - 1:
                            end_idx = idx_num
                        else:
                            end_idx = start_idx + cut_idx

                        self.lanelets[id_]['cut_idx'].append(
                            [start_idx, end_idx])  # 절단 인덱스 추가

                else:
                    # 두 번째 이후 레인렛의 절단 인덱스 설정
                    for i in range(cut_num):
                        pre_id = group[n-1]  # 이전 레인렛 ID
                        pre_end_idx = self.lanelets[pre_id]['cut_idx'][i][1] - 1  # 이전 레인렛의 끝 인덱스

                        pt = self.lanelets[pre_id]['waypoints'][pre_end_idx]  # 이전 레인렛의 끝점

                        if i == 0:
                            start_idx = 0
                            end_idx = find_nearest_idx(
                                self.lanelets[id_]['waypoints'], pt)  # 가장 가까운 인덱스 찾기

                        elif i == cut_num - 1:
                            start_idx = self.lanelets[id_]['cut_idx'][i-1][1]
                            end_idx = self.lanelets[id_]['idx_num']

                        else:
                            start_idx = self.lanelets[id_]['cut_idx'][i-1][1]
                            end_idx = find_nearest_idx(
                                self.lanelets[id_]['waypoints'], pt)

                        self.lanelets[id_]['cut_idx'].append(
                            [start_idx, end_idx])  # 절단 인덱스 추가

        # 그래프 연결 설정
        for id_, data in self.lanelets.items():
            if data['group'] is None:
                if self.graph.get(id_) is None:
                    self.graph[id_] = {}

                for p_id in data['successor']:
                    if self.lanelets[p_id]['group'] is None:
                        self.graph[id_][p_id] = data['length']
                    else:
                        self.graph[id_][p_id+'_0'] = data['length']

            else:
                last = len(data['cut_idx']) - 1
                for n in range(len(data['cut_idx'])):
                    new_id = '%s_%s' % (id_, n)
                    if self.graph.get(new_id) is None:
                        self.graph[new_id] = {}

                    if n == last:
                        for p_id in data['successor']:
                            if self.lanelets[p_id]['group'] is None:
                                self.graph[new_id][p_id] = self.cut_dist
                            else:
                                self.graph[new_id][p_id+'_0'] = self.cut_dist

                    else:
                        self.graph[new_id]['%s_%s' %
                                           (id_, n+1)] = self.cut_dist

                        s_idx, e_idx = self.lanelets[id_]['cut_idx'][n]

                        left_id = data['adjacentLeft']
                        if left_id is not None:
                            if sum(self.lanelets[id_]['leftChange'][s_idx:s_idx+(e_idx-s_idx)//2]) == (e_idx - s_idx)//2:
                                self.graph[new_id]['%s_%s' % (
                                    left_id, n+1)] = self.cut_dist + 10.0 + n * 0.1

                        right_id = data['adjacentRight']
                        if right_id is not None:
                            if sum(self.lanelets[id_]['rightChange'][s_idx:s_idx+(e_idx-s_idx)//2]) == (e_idx - s_idx)//2:
                                self.graph[new_id]['%s_%s' % (
                                    right_id, n+1)] = self.cut_dist + 10.0 + n * 0.1

        # 역방향 그래프 생성
        self.reversed_graph = {}
        for from_id, data in self.graph.items():
            for to_id in data:
                if self.reversed_graph.get(to_id) is None:
                    self.reversed_graph[to_id] = []

                self.reversed_graph[to_id].append(from_id)

