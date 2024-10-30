#ifndef GYR_INT_H
#define GYR_INT_H

#include <sensor_msgs/Imu.h>
#include <novatel_oem7_msgs/INSPVA.h>
#include "sophus/so3.hpp"

class GyrInt 
{
  public:
  GyrInt();
  void Integrate(const sensor_msgs::ImuConstPtr &imu);
  void IntegrateFromInspva(const novatel_oem7_msgs::INSPVAConstPtr &inspva); // 추가된 메서드
  void Reset(double start_timestamp, const sensor_msgs::ImuConstPtr &lastimu);

  const Sophus::SO3d GetRot() const;

  private:
  double start_timestamp_;
  sensor_msgs::ImuConstPtr last_imu_;
  std::vector<sensor_msgs::ImuConstPtr> v_imu_;
  std::vector<Sophus::SO3d> v_rot_;
};

#endif 
