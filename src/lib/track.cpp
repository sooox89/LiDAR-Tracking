#include "track/track.h"
#include "utils.hpp"
#include <cmath>

using namespace cv;

//constructor
Track::Track()
{
	stateVariableDim = 5;           // cx, cy, yaw, dx, dy
	stateMeasureDim = 3;            // cx, cy, yaw
	nextID = 0;
	m_thres_invisibleCnt = 10;
	n_velocity_deque = 6;
	n_orientation_deque = 6;
	thr_velocity = 15; // m/s
	// thr_orientation = M_PI / 6; // 30 degree -> rad : 0.5236
	thr_orientation = 0.5236; // 30 degree -> rad : 0.5236

	//A & Q ==> Predict process
	//H & R ==> Estimation process
	
	// A 상태
	m_matTransition = Mat::eye(stateVariableDim, stateVariableDim, CV_32F);
	m_matTransition.at<float>(0, 3) = dt;
	m_matTransition.at<float>(1, 4) = dt;

	// H 관측
	m_matMeasurement = Mat::zeros(stateMeasureDim, stateVariableDim, CV_32F);
	m_matMeasurement.at<float>(0, 0) = 1.0f; // cx
	m_matMeasurement.at<float>(1, 1) = 1.0f; // cy
	m_matMeasurement.at<float>(2, 2) = 1.0f; // yaw
	
	// Q System Noise : 클수록 Kalman Gain이 증가함, 측정값의 영향 증가
	float Q[] = {1e-1f, 1e-1f, 1e-2f, 1e-2f, 1e-2f};
	Mat tempQ(stateVariableDim, 1, CV_32FC1, Q);
	m_matProcessNoiseCov = Mat::diag(tempQ);

    // R Measurment Noise : 클수록 Kalman Gain이 감소, 측정값의 영향 감소
	float R[] = {1e-2f, 1e-2f, 1e-1f};
	Mat tempR(stateMeasureDim, 1, CV_32FC1, R);
	m_matMeasureNoiseCov = Mat::diag(tempR);

	m_thres_associationCost = 5.0f;
}

//deconstructor
Track::~Track(){}

void Track::velocity_push_back(std::deque<float> &deque, float v) 
{
	if (deque.size() < n_velocity_deque) { deque.push_back(v); }
	else { // 제로백 5초 기준 가속도는 5.556m/(s*)
		float sum_vel = 0.0;
        for (size_t i = 0; i < deque.size(); ++i) { sum_vel += deque[i]; }
        float avg_vel = sum_vel / deque.size();
		
		if (abs(avg_vel - v) < thr_velocity) { deque.push_back(v); }
		else { deque.push_back(deque.back()); }
		deque.pop_front(); }
}

void Track::orientation_push_back(std::deque<float> &deque, float o)
{
    if (deque.size() < n_orientation_deque) { deque.push_back(o); }
    else {
        float sum_yaw = 0.0;
        for (size_t i = 0; i < deque.size(); ++i) { sum_yaw += deque[i]; }

        float avg_yaw = sum_yaw / deque.size();
        float yaw_diff = std::fabs(angles::shortest_angular_distance(avg_yaw, o));
        if (yaw_diff <= 0.5236) { deque.push_back(o); }
        else 
		{ 
			std::cout << std::endl;
			deque.push_back(deque.back()); 
		}

		deque.pop_front();
    }
}

float Track::getVectorScale(float v1, float v2)
{
	float distance = sqrt(pow(v1, 2) + pow(v2, 2));
	if (v1 < 0) return -distance;
	else return distance;
}

double Track::getBBoxRatio(jsk_recognition_msgs::BoundingBox bbox1, jsk_recognition_msgs::BoundingBox bbox2)
{
	double boxA[4] = {bbox1.pose.position.x - bbox1.dimensions.x/2.0, 
					 bbox1.pose.position.y - bbox1.dimensions.y/2.0, 
					 bbox1.pose.position.x + bbox1.dimensions.x/2.0, 
					 bbox1.pose.position.y + bbox1.dimensions.y/2.0};
 	double boxB[4] = {bbox2.pose.position.x - bbox2.dimensions.x/2.0, 
					 bbox2.pose.position.y - bbox2.dimensions.y/2.0, 
					 bbox2.pose.position.x + bbox2.dimensions.x/2.0, 
					 bbox2.pose.position.y + bbox2.dimensions.y/2.0};
 	
 	double boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
	double boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]);

	double gap;
	if(boxAArea > boxBArea) gap = boxAArea / boxBArea;
	else gap = boxBArea / boxAArea;

	return gap;
}

double Track::getBBoxDistance(jsk_recognition_msgs::BoundingBox bbox1, jsk_recognition_msgs::BoundingBox bbox2)
{	
	float distance = sqrt(pow(bbox2.pose.position.x - bbox1.pose.position.x, 2) + pow(bbox2.pose.position.y - bbox1.pose.position.y, 2));
	return distance;
}

bool Track::has_recent_values_same_sign(const std::deque<float>& dq, int n)
{
    if (dq.size() < n) return false;

    int last_sign = (dq.back() >= 0) ? 1 : -1;

    for (int i = dq.size() - n; i < dq.size(); ++i)
    {
        int sign = (dq[i] >= 0) ? 1 : -1;
        if (sign != last_sign)
        {
            return false;
        }
    }
    return true;
}

visualization_msgs::Marker Track::get_text_msg(struct trackingStruct &track, int i)
{
	visualization_msgs::Marker text;
	text.ns = "text";
	text.id = i;
	text.action = visualization_msgs::Marker::ADD;
	text.lifetime = ros::Duration(0.1);
	text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
	text.color.r = 1.0;
	text.color.g = 1.0;
	text.color.b = 1.0;
	text.color.a = 1.0;
	text.scale.z = 1.0;

	text.pose.position.x = track.cur_bbox.pose.position.x;
	text.pose.position.y = track.cur_bbox.pose.position.y;
	text.pose.position.z = track.cur_bbox.pose.position.z + 1.5;
	text.pose.orientation.w = 1.0;

	char buf[100];
	//sprintf(buf, "ID : %d", track.id*10+1);
	//sprintf(buf, "%f", text.pose.position.y);
	// sprintf(buf, "ID: %d\nAge: %d\nV: %dkm/h", track.id, track.age ,int(track.v*3.6));
	sprintf(buf, "Age: %d\nV: %dkm/h", track.age, int(track.v*3.6));

	text.text = buf;

	return text;
}

void Track::predictNewLocationOfTracks(const ros::Time &currentTime)
{
    for (int i = 0; i < vecTracks.size(); i++)
    {

		// Tracking 객체에 대한 dt 계산
        dt = currentTime.toSec() - vecTracks[i].lastTime;

        // 상태 전이 행렬 업데이트 (등가속도 모델 반영)
		// A Matrix
        vecTracks[i].kf.transitionMatrix = Mat::eye(stateVariableDim, stateVariableDim, CV_32F);
        vecTracks[i].kf.transitionMatrix.at<float>(0, 3) = dt;
        vecTracks[i].kf.transitionMatrix.at<float>(1, 4) = dt;

        // 상태 예측
        vecTracks[i].kf.predict();

        // 예측된 위치와 속도 업데이트
        vecTracks[i].cur_bbox.pose.position.x = vecTracks[i].kf.statePre.at<float>(0);
        vecTracks[i].cur_bbox.pose.position.y = vecTracks[i].kf.statePre.at<float>(1);
        vecTracks[i].orientation = vecTracks[i].kf.statePre.at<float>(2);
        vecTracks[i].vx = vecTracks[i].kf.statePre.at<float>(3);
        vecTracks[i].vy = vecTracks[i].kf.statePre.at<float>(4);
    }
}

void Track::assignDetectionsTracks(const jsk_recognition_msgs::BoundingBoxArray &bboxArray)
{
	int N = (int)vecTracks.size();       //  N = number of tracking
	int M = (int)bboxArray.boxes.size(); //  M = number of detection

	vector<vector<double>> Cost(N, vector<double>(M)); //2 array

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < M; j++)
		{
			// 각 차량에 대한 예측값과 실제 검출된 객체와의 Center Distance 및 IoU를 고려한 cost 생성
			Cost[i][j] = f32_weight_dist * getBBoxDistance(vecTracks[i].cur_bbox, bboxArray.boxes[j])
							+ f32_weight_iou * (1-getBBoxOverlap(vecTracks[i].cur_bbox, bboxArray.boxes[j]));
		}
	}

	vector<int> assignment;

	// Tracking 객체에 대해 현재 State에서 Matching 된 Object의 Index가 assignment[i]에 할당됨
	if (N != 0)
	{
		AssignmentProblemSolver APS;
		APS.Solve(Cost, assignment, AssignmentProblemSolver::optimal);
	}

	vecAssignments.clear();
	vecUnassignedTracks.clear();
	vecUnssignedDetections.clear();

	for (int i = 0; i < N; i++)
	{
		if (assignment[i] == -1)	// Matching 되지 않은 경우
		{
			vecUnassignedTracks.push_back(i);
		}
		else
		{
			// Tracking Object가 Matching 되는 경우 
			// 두 객체 사이의 거리가 m_thres_associationCost 보다 작은 경우 최종적으로 Matching 됨을 판단
			std::cout<<"Kalman: "<<vecTracks[i].kf.statePost.at<float>(2)<<", Measurement: "<<tf::getYaw(bboxArray.boxes[assignment[i]].pose.orientation)<<std::endl;
			if (Cost[i][assignment[i]] < m_thres_associationCost && abs(vecTracks[i].kf.statePost.at<float>(2)-tf::getYaw(bboxArray.boxes[assignment[i]].pose.orientation)) < 0.5236)
			{
				vecAssignments.push_back(pair<int, int>(i, assignment[i]));
			}
			else
			{
				vecUnassignedTracks.push_back(i);
				assignment[i] = -1;	// Matching 되지 않았다고 설정
			}
		}
	}

	for (int j = 0; j < M; j++)
	{
		auto it = find(assignment.begin(), assignment.end(), j);
		if (it == assignment.end())
			vecUnssignedDetections.push_back(j);
	}
}

// 상대 좌표계
void Track::assignedTracksUpdate(const jsk_recognition_msgs::BoundingBoxArray &bboxArray)
{	
	for (int i = 0; i < (int)vecAssignments.size(); i++)
	{
		int idT = vecAssignments[i].first;		// Tracking
		int idD = vecAssignments[i].second;		// Detection

		// 예측된 Tracking 객체와 실제 검출한 객체와의 Center 값의 차이와 dt를 이용해 vx, vy 계산
		float dx = bboxArray.boxes[idD].pose.position.x - vecTracks[idT].pre_bbox.pose.position.x;
		float dy = bboxArray.boxes[idD].pose.position.y - vecTracks[idT].pre_bbox.pose.position.y;

        float vx = dx / dt;
        float vy = dy / dt;
        
		// Moving Average Filter를 통한 vx, vy 계산 안정화
        velocity_push_back(vecTracks[idT].vx_deque, vx);
        velocity_push_back(vecTracks[idT].vy_deque, vy);

        vecTracks[idT].vx = std::accumulate(vecTracks[idT].vx_deque.begin(), vecTracks[idT].vx_deque.end(), 0.0f) / vecTracks[idT].vx_deque.size();
        vecTracks[idT].vy = std::accumulate(vecTracks[idT].vy_deque.begin(), vecTracks[idT].vy_deque.end(), 0.0f) / vecTracks[idT].vy_deque.size();
		
        Mat measure = Mat::zeros(stateMeasureDim, 1, CV_32FC1);
        measure.at<float>(0) = bboxArray.boxes[idD].pose.position.x;
        measure.at<float>(1) = bboxArray.boxes[idD].pose.position.y;

        vecTracks[idT].kf.correct(measure);

		// Kalman Filter 끝
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------


		// cout<<idT<<"'s Kalman Orient: "<<vecTracks[idT].kf.statePost.at<float>(2)<<", Using Orientation: "<<vecTracks[idT].orientation<<std::endl;
		vecTracks[idT].v = getVectorScale(vecTracks[idT].vx, vecTracks[idT].vy);

		// 이전 orientation들과 비교
        orientation_push_back(vecTracks[idT].orientation_deque, tf::getYaw(bboxArray.boxes[idD].pose.orientation));
		// std::cout << "할당된 Yaw: "<<vecTracks[idT].orientation_deque.back()<<std::endl;
		vecTracks[idT].cur_bbox.pose.orientation = tf::createQuaternionMsgFromYaw(vecTracks[idT].orientation_deque.back());

		if (bboxArray.boxes[idD].label == 0 && vecTracks[idT].pre_bbox.label != 0 && vecTracks[idT].age > m_thres_invisibleCnt) {
			vecTracks[idT].cur_bbox.dimensions = vecTracks[idT].pre_bbox.dimensions;
			vecTracks[idT].cur_bbox.pose.orientation = vecTracks[idT].pre_bbox.pose.orientation; 
			vecTracks[idT].cur_bbox.label = vecTracks[idT].pre_bbox.label; 
			}
		else {
			
			tf2::Quaternion quat_tf, quat_tf2;
			tf2::fromMsg(vecTracks[idT].pre_bbox.pose.orientation, quat_tf);

			// 오일러 각으로 변환
			double roll, pitch, yaw;
			double roll_after, pitch_after, yaw_after;
			tf2::Matrix3x3(quat_tf).getRPY(roll, pitch, yaw);

			tf2::fromMsg(vecTracks[idT].cur_bbox.pose.orientation, quat_tf2);
			tf2::Matrix3x3(quat_tf2).getRPY(roll_after, pitch_after, yaw_after);

			if(abs(yaw_after-yaw)>0.5236)
			{
				vecTracks[idT].cur_bbox.dimensions = vecTracks[idT].pre_bbox.dimensions;
				vecTracks[idT].cur_bbox.pose.orientation = vecTracks[idT].pre_bbox.pose.orientation; 
				vecTracks[idT].cur_bbox.label = vecTracks[idT].pre_bbox.label; 

				tf2::fromMsg(vecTracks[idT].pre_bbox.pose.orientation, quat_tf);
				tf2::Matrix3x3(quat_tf).getRPY(roll, pitch, yaw);

				tf2::fromMsg(vecTracks[idT].cur_bbox.pose.orientation, quat_tf2);
				tf2::Matrix3x3(quat_tf2).getRPY(roll_after, pitch_after, yaw_after);

				// 처음에 Yaw가 잘못 검출되는 경우를 방지하기 위해
				if(vecTracks[idT].f_continue_misoriented == false)
				{
					vecTracks[idT].cnt_misoriented += 1;
					vecTracks[idT].f_continue_misoriented = true;
				}
				else
				{
					vecTracks[idT].cnt_misoriented += 1;
					if(vecTracks[idT].cnt_misoriented > 3)
					{
						vecTracks[idT].cnt_misoriented = 0;
						vecTracks[idT].f_continue_misoriented = false;
						vecTracks[idT].cur_bbox.pose.orientation = bboxArray.boxes[idD].pose.orientation;
						vecTracks[idT].orientation_deque.clear();
					}
				}
			}
			else{
				vecTracks[idT].cur_bbox.dimensions = bboxArray.boxes[idD].dimensions;
				// vecTracks[idT].cur_bbox.pose.orientation = bboxArray.boxes[idD].pose.orientation;
				vecTracks[idT].cur_bbox.label = bboxArray.boxes[idD].label; 


			}
			}


		// Kalman Yaw로 사용

		// Roll, Pitch는 0이고, Yaw만 존재
		tf::Quaternion quat_tf = tf::createQuaternionFromRPY(0.0, 0.0, vecTracks[idT].kf.statePost.at<float>(2));

		// tf → geometry_msgs::Quaternion 변환
		geometry_msgs::Quaternion quat_msg;
		tf::quaternionTFToMsg(quat_tf, quat_msg);
		vecTracks[idT].cur_bbox.pose.orientation = quat_msg;
		// ------------------------------------------------------------------------------------
		vecTracks[idT].cur_bbox.pose.position = bboxArray.boxes[idD].pose.position;
		// vecTracks[idT].cur_bbox.label = bboxArray.boxes[idD].label;
		vecTracks[idT].cur_bbox.header.stamp = bboxArray.boxes[idD].header.stamp; // 중요, 이거 안 하면 time stamp 안 맞아서 스탬프가 오래됐을 경우 rviz에서 제대로 표시 안됨

		vecTracks[idT].pre_bbox = vecTracks[idT].cur_bbox;
		vecTracks[idT].cls = vecTracks[idT].cur_bbox.label;
		if (vecTracks[idT].cls != 0) { vecTracks[idT].score = bboxArray.boxes[idD].value; }
		vecTracks[idT].lastTime = bboxArray.boxes[idD].header.stamp.toSec();
		vecTracks[idT].age++;
		vecTracks[idT].cntTotalVisible++;
		vecTracks[idT].cntConsecutiveInvisible = 0;
		// vecTracks[idT].f_consecutive_invisible = false;
	}
}

void Track::unassignedTracksUpdate()
{
	for (int i = 0; i < (int)vecUnassignedTracks.size(); i++)
	{
		int id = vecUnassignedTracks[i];
		vecTracks[id].age++;
		vecTracks[id].cntConsecutiveInvisible++;

		// 객체 유지
		vecTracks[id].cur_bbox = vecTracks[id].pre_bbox;
	}
}

void Track::deleteLostTracks()
{
	if ((int)vecTracks.size() == 0)
	{
		return;
	}

	for (int i = vecTracks.size() - 1; i >= 0; i--)
	{
		if (vecTracks[i].cntConsecutiveInvisible >= m_thres_invisibleCnt)
		{
			vecTracks.erase(vecTracks.begin() + i);
		}
	}

}

void Track::createNewTracks(const jsk_recognition_msgs::BoundingBoxArray &bboxArray)
{
	for (int i = 0; i < (int)vecUnssignedDetections.size(); i++)
	{
		int id = vecUnssignedDetections[i];

		trackingStruct ts;
		ts.id = nextID++;
		ts.age = 1;
		ts.cntTotalVisible = 1;
		ts.cntConsecutiveInvisible = 0;

		ts.cur_bbox = bboxArray.boxes[id];
		ts.pre_bbox = bboxArray.boxes[id];

		ts.vx = 0.0;
		ts.vy = 0.0;
		ts.v = 0.0;

		ts.kf.init(stateVariableDim, stateMeasureDim);

		m_matTransition.copyTo(ts.kf.transitionMatrix);         //A
		m_matMeasurement.copyTo(ts.kf.measurementMatrix);       //H
		m_matProcessNoiseCov.copyTo(ts.kf.processNoiseCov);     //Q
		m_matMeasureNoiseCov.copyTo(ts.kf.measurementNoiseCov); //R

		// 오차 공분산 초기값
		float P[] = {1e-2f, 1e-2f, 1e-1f, 1e-1f, 1e-1f};
		Mat tempCov(stateVariableDim, 1, CV_32FC1, P);
		ts.kf.errorCovPost = Mat::diag(tempCov);

		ts.kf.statePost.at<float>(0) = ts.cur_bbox.pose.position.x;
		ts.kf.statePost.at<float>(1) = ts.cur_bbox.pose.position.y;
		ts.kf.statePost.at<float>(2) = tf::getYaw(ts.cur_bbox.pose.orientation);
		ts.kf.statePost.at<float>(3) = ts.vx;
		ts.kf.statePost.at<float>(4) = ts.vy;

		ts.lastTime = bboxArray.boxes[id].header.stamp.toSec();
		
		vecTracks.push_back(ts);
	}
}

pair<jsk_recognition_msgs::BoundingBoxArray, visualization_msgs::MarkerArray> Track::displayTrack()
{   
	jsk_recognition_msgs::BoundingBoxArray bboxArray;
	visualization_msgs::MarkerArray textArray;
	std::cout << vecTracks.size() << std::endl;
	for (int i = 0; i < vecTracks.size(); i++)
	{
		// std::cout<<i<<"'s Age: "<<vecTracks[i].age<<", cntConsecutiveInvisible: "<<vecTracks[i].cntConsecutiveInvisible<<std::endl;
		if (vecTracks[i].age >= 1 && vecTracks[i].cntConsecutiveInvisible == 0)
		// if (vecTracks[i].age >= 1)
		{	
			vecTracks[i].cur_bbox.header.seq = vecTracks[i].age; // header.seq를 tracking object의 age로 사용
			vecTracks[i].cur_bbox.value = vecTracks[i].v;			
			bboxArray.boxes.push_back(vecTracks[i].cur_bbox);
			textArray.markers.push_back(get_text_msg(vecTracks[i], i));
		}
	}
	// std::cout << "After :" << bboxArray.boxes.size() << std::endl;
	
	pair<jsk_recognition_msgs::BoundingBoxArray, visualization_msgs::MarkerArray> bbox_marker(bboxArray, textArray);
	return bbox_marker;
}
