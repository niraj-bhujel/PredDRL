
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "ros/ros.h"
#include <ros/console.h>
#include <nav_msgs/Path.h>
#include <std_msgs/String.h>
#include <math.h>

#include <nav_msgs/Odometry.h>


#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf/transform_broadcaster.h>
#include <tf/tf.h>


nav_msgs::Path  path;

ros::Publisher  path_pub;
ros::Subscriber odomSub;
ros::Subscriber odom_raw_Sub;

float length = 0;
float last_x=0,last_y=0;

 void odomCallback(const nav_msgs::Odometry::ConstPtr& odom)
 {
    geometry_msgs::PoseStamped this_pose_stamped;
    this_pose_stamped.pose.position.x = odom->pose.pose.position.x;
    this_pose_stamped.pose.position.y = odom->pose.pose.position.y;

    this_pose_stamped.pose.orientation = odom->pose.pose.orientation;

    this_pose_stamped.header.stamp = ros::Time::now();
    this_pose_stamped.header.frame_id = "odom";

    path.poses.push_back(this_pose_stamped);

    path.header.stamp = ros::Time::now();
    path.header.frame_id="odom";
    path_pub.publish(path);


    length += sqrt(pow(odom->pose.pose.position.x-last_x,2)+pow(odom->pose.pose.position.y-last_y,2));
    last_x = odom->pose.pose.position.x;
    last_y = odom->pose.pose.position.y;
    // ROS_INFO("total length: %f", length);

//    ROS_INFO("odom %.3lf %.3lf",odom->pose.pose.position.x,odom->pose.pose.position.y);
 }


int main (int argc, char **argv)
{
    ros::init (argc, argv, "odomtopath");

    ros::NodeHandle ph;

    path_pub = ph.advertise<nav_msgs::Path>("trajectory",1, true);
    odomSub  = ph.subscribe<nav_msgs::Odometry>("/odom", 1, odomCallback);

    ros::Rate loop_rate(20);
    while (ros::ok())
    {
        ros::spinOnce();               // check for incoming messages
        loop_rate.sleep();
    }

    return 0;
}