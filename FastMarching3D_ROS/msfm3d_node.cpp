// g++ msfm3d_node.cpp -g -o msfm3d_node.o -I /opt/ros/melodic/include -I /usr/include/c++/7.3.0 -I /home/andrew/catkin_ws/devel/include -I /home/andrew/catkin_ws/src/octomap_msgs/include -I /usr/include/pcl-1.8 -I /usr/include/eigen3 -L /usr/lib/x86_64-linux-gnu -L /home/andrew/catkin_ws/devel/lib -L /opt/ros/melodic/lib -Wl,-rpath,opt/ros/melodic/lib -lroscpp -lrosconsole -lrostime -lroscpp_serialization -loctomap -lboost_system -lpcl_common -lpcl_io -lpcl_filters -lpcl_features -lpcl_kdtree -lpcl_segmentation

#include <math.h>
#include <numeric>
#include <algorithm>
#include <random>
// ROS libraries
#include <ros/ros.h>
#include <std_msgs/String.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Int8.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Float32MultiArray.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/MarkerArray.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <msfm3d/Goal.h>
#include <msfm3d/GoalArray.h>
// #include <marble_common/ArtifactArray.msg>
// Octomap libaries
#include <octomap/octomap.h>
#include <octomap/ColorOcTree.h>
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/conversions.h>
// pcl libraries
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/frustum_culling.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
//pcl ROS
#include <pcl_conversions/pcl_conversions.h>
// Custom libraries
#include "msfm3d.c"
#include "bresenham3d.cpp"

template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T> &v) {

  // initialize original index locations
  std::vector<size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  std::sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  return idx;
}

// Converts 4 integer byte values (0 to 255) to a float
float byte2Float(const int a[4])
{
  // Declare union holder variable for byte2float conversion
  union {
    unsigned char bytes[4];
    float f;
  } byteFloat;
  // Store values in a in union variable.
  for (int i = 0; i < 4; i++) {
    byteFloat.bytes[i] = a[i];
  }
  return byteFloat.f;
}

float dist2(const float a[2], const float b[2]){
  float sum = 0.0;
  for (int i=0; i < 2; i++) sum += (b[i] - a[i])*(b[i] - a[i]);
  return std::sqrt(sum);
}

float dist3(const float a[3], const float b[3]){
  float sum = 0.0;
  for (int i=0; i < 3; i++) sum += (b[i] - a[i])*(b[i] - a[i]);
  return std::sqrt(sum);
}

int sign(float a){
  if (a>0.0) return 1;
  if (a<0.0) return -1;
  return 0;
}

float angle_diff(float a, float b)
{
    // Computes a-b, preserving the correct sign (counter-clockwise positive angles)
    // All angles are in degrees
    a = std::fmod(360000.0 + a, 360.0);
    b = std::fmod(360000.0 + b, 360.0);
    float d = a - b;
    d = std::fmod(d + 180.0, 360.0) - 180.0;
    return d;
}

// Some orientation and pose structures
struct Quaternion {
  float w, x, y, z;
};

Quaternion euler2Quaternion(float yaw, float pitch, float roll) // yaw (Z), pitch (Y), roll (X)
{
  // From Wikipedia
  // Abbreviations for the various angular functions
  float cy = std::cos(yaw * 0.5);
  float sy = std::sin(yaw * 0.5);
  float cp = std::cos(pitch * 0.5);
  float sp = std::sin(pitch * 0.5);
  float cr = std::cos(roll * 0.5);
  float sr = std::sin(roll * 0.5);

  Quaternion q;
  q.w = cy * cp * cr + sy * sp * sr;
  q.x = cy * cp * sr - sy * sp * cr;
  q.y = sy * cp * sr + cy * sp * cr;
  q.z = sy * cp * cr - cy * sp * sr;
  return q;
}

struct Pose {
  pcl::PointXYZ position;
  Eigen::Matrix3f R;
  Quaternion q;
};
struct SensorFoV {
  float verticalFoV; // in degrees
  float horizontalFoV; // in degrees
  float rMin; // in meters
  float rMax; // in meters
};
struct View {
  Pose pose;
  pcl::PointCloud<pcl::PointXYZ> cloud;
  int index = -1;
};

//Msfm3d class declaration
class Msfm3d
{
  public:
    // Constructor
    Msfm3d(float map_resolution):
    frontierCloud(new pcl::PointCloud<pcl::PointXYZ>)
    {
      reach = NULL;
      esdf.data = NULL;
      esdf.seen = NULL;
      frontier = NULL;
      entrance = NULL;
      voxel_size = map_resolution;
      mytree = new octomap::OcTree(voxel_size);
    }

    // Structure definitions
    struct ESDF {
      double * data; // esdf matrix pointer
      bool * seen; // seen matrix pointer
      int size[3]; // number of elements in each dimension
      float max[4]; // max and min values in each dimension
      float min[4];
    };
    struct Boundary {
      bool set = 0;
      float xmin = 0.0, xmax = 0.0, ymin = 0.0, ymax = 0.0, zmin = 0.0, zmax = 0.0;
    };

    // Vehicle parameters
    bool ground = false; // whether the vehicle is a ground vehicle
    bool fixGoalHeightAGL = false; // whether or not the vehicle has a fixed goal point height above ground level
    float goalHeightAGL = 0.64; // meters
    float position[3] = {69.0, 420.0, 1337.0}; // robot position
    float euler[3]; // robot orientation in euler angles
    float R[9]; // Rotation matrix
    Boundary vehicleVolume; // xyz boundary of the vehicle bounding box in rectilinear coordinates for collision detection/avoidance

    // Vehicle linear and angular velocities
    float speed = 1.0; // m/s
    float turnPenalty = 0.2; // Weight to place on an initial heading error

    // Environment/Sensor parameters
    std::string frame = "world";
    bool esdf_or_octomap = 0; // Boolean to use an esdf PointCloud2 or an Octomap as input
    bool receivedPosition = 0;
    bool receivedMap = 0;
    bool updatedMap = 0;
    float voxel_size;
    float bubble_radius = 1.0; // map voxel size, and bubble radius
    float origin[3]; // location in xyz coordinates where the robot entered the environment
    float entranceRadius; // radius around the origin where frontiers can't exist
    float inflateWidth = 0.0;
    int minViewCloudSize = 10;

    float viewPoseObstacleDistance = 0.001; // view pose minimum distance from obstacles

    double * reach; // reachability grid (output from reach())
    sensor_msgs::PointCloud2 PC2msg;
    nav_msgs::Path pathmsg;
    // visualization_msgs::MarkerArray frontiermsg;
    octomap::OcTree* mytree; // OcTree object for holding Octomap

    // Frontier and frontier filter parameters
    // std::vector<bool> frontier;
    // float
    bool * frontier;
    bool * entrance;
    int frontier_size = 0;
    int dzFrontierVoxelWidth = 0;
      // Clustering/Filtering
      pcl::PointCloud<pcl::PointXYZ>::Ptr frontierCloud; // Frontier PCL
      std::vector<pcl::PointIndices> frontierClusterIndices;
      float cluster_radius;
      float min_cluster_size;
      float normalThresholdZ;
      // ROS Interfacing
      sensor_msgs::PointCloud2 frontiermsg;
      // Frontier Grouping
      std::vector<pcl::PointIndices> greedyGroups;
      pcl::PointCloud<pcl::PointXYZ> greedyCenters;
      std::vector<int> greedyClusterNumber;

    ESDF esdf; // ESDF struct object
    Boundary bounds; // xyz boundary of possible goal locations

    void callback(sensor_msgs::PointCloud2 msg); // Subscriber callback function for PC2 msg (ESDF)
    void callback_Octomap(const octomap_msgs::Octomap::ConstPtr msg); // Subscriber callback function for Octomap msg
    void callback_Octomap_freePCL(const sensor_msgs::PointCloud2 msg);
    void callback_Octomap_occupiedPCL(const sensor_msgs::PointCloud2 msg);
    void callback_position(const nav_msgs::Odometry msg); // Subscriber callback for robot position
    void parsePointCloud(); // Function to parse pointCloud2 into an esdf format that msfm3d can use
    int xyz_index3(const float point[3]);
    void index3_xyz(const int index, float point[3]);
    void getEuler(); // Updates euler array given the current quaternion values
    bool updatePath(const float goal[3]); // Updates the path vector from the goal frontier point to the robot location
    void updateFrontierMsg(); // Updates the frontiermsg MarkerArray with the frontier matrix for publishing
    bool clusterFrontier(const bool print2File); // Clusters the frontier pointCloud with euclidean distance within a radius
    bool normalFrontierFilter(); // Filters frontiers based on they're local normal value
    bool inBoundary(const float point[3]); // Checks if a point is inside the planner boundaries
    void greedyGrouping(const float r, const bool print2File);
};

bool Msfm3d::inBoundary(const float point[3])
{
  if (bounds.set){
    if ((bounds.xmin <= point[0]) && (bounds.xmax >= point[0]) && (bounds.ymin <= point[1]) && (bounds.ymax >= point[1]) && (bounds.zmin <= point[2]) && (bounds.zmax >= point[2])) {
      return 1;
    } else {
      return 0;
    }
  } else {
    // No boundary exists so every point is inside.
    return 1;
  }
}

bool Msfm3d::clusterFrontier(const bool print2File)
{
  // clusterFrontier takes as input the frontierCloud and the voxel_size and extracts contiguous clusters of minimum size round(12.0/voxel_size) (60 for voxel_size=0.2) voxels.
  // The indices of the cluster members are stored in frontierClusterIndices and the unclustered voxels are filtered from frontierCloud.
  // Input:
  //    - Msfm3d.frontierCloud (pcl::PointCloud<pcl::PointXYZ>::Ptr)
  //    - Msfm3d.voxel_size (float)
  // Output:
  //    - Msfm3d.frontierClusterIndices (std::vector<pcl::PointIndices>)
  //    - Msfm3d.frontierCloud (pcl::PointCloud<pcl::PointXYZ>::Ptr)
  //
  // Lines commented out include debugging ROS_INFO() text and .pcd file storage of the frontier cluster PointClouds.

  ROS_INFO("Frontier cloud before clustering has: %d data points.", (int)frontierCloud->points.size());

  // Create cloud pointer to store the removed points
  pcl::PointCloud<pcl::PointXYZ>::Ptr removed_points(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr frontierCloudPreFilter(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
  pcl::copyPointCloud(*frontierCloud, *frontierCloudPreFilter);

  // Creating the KdTree object for the search method of the extraction
  pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
  kdtree->setInputCloud(frontierCloud);

  // Clear previous frontierClusterIndices
  frontierClusterIndices.clear();

  // Initialize euclidean cluster extraction object
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance(cluster_radius); // Clusters must be made of contiguous sections of frontier (within sqrt(2)*voxel_size of each other)
  ec.setMinClusterSize(roundf(min_cluster_size)); // Cluster must be at least 15 voxels in size
  ec.setSearchMethod(kdtree);
  ec.setInputCloud(frontierCloud);
  ec.extract(frontierClusterIndices);

  // Iterate through clusters and write to file
  int j = 0;
  pcl::PCDWriter writer;
  for (std::vector<pcl::PointIndices>::const_iterator it = frontierClusterIndices.begin(); it != frontierClusterIndices.end(); ++it){
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
    for (std::vector<int>::const_iterator pit=it->indices.begin(); pit!=it->indices.end(); ++pit){
      if (print2File) cloud_cluster->points.push_back(frontierCloud->points[*pit]);
      inliers->indices.push_back(*pit); // Indices to keep in frontierCloud
    }
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid (*cloud_cluster, centroid);
    ROS_WARN("CLUSTER CENTROID: x = %f, y = %f, z = %f", centroid[0], centroid[1], centroid[2]);

    
  }

  // Loop through the remaining points in frontierCloud and remove them from the frontier bool array
  pcl::ExtractIndices<pcl::PointXYZ> extract;
  extract.setInputCloud(frontierCloud);
  extract.setIndices(inliers);
  extract.setNegative(true);
  extract.filter(*removed_points);
  float query[3];
  int idx;
  for (pcl::PointCloud<pcl::PointXYZ>::iterator it=removed_points->begin(); it!=removed_points->end(); ++it){
    query[0] = it->x;
    query[1] = it->y;
    query[2] = it->z;
    idx = xyz_index3(query);
    frontier[idx] = 0;
    // ROS_INFO("Removed frontier voxel at (%f, %f, %f).", query[0], query[1], query[2]);
  }

  // Filter frontierCloud to keep only the inliers
  frontierCloud->clear();
  idx = 0;
  for (int i=0; i<frontierClusterIndices.size(); i++) {
    for (int j=0; j<frontierClusterIndices[i].indices.size(); j++) {
      frontierCloud->points.push_back(frontierCloudPreFilter->points[frontierClusterIndices[i].indices[j]]);
      frontierClusterIndices[i].indices[j] = idx;
      idx++;
    }
  }
  ROS_INFO("Frontier cloud after clustering has %d points.", (int)frontierCloud->points.size());

  if ((int)frontierCloud->points.size() < 1) {
    return 0;
  } else {
    // Get new indices of Frontier Clusters after filtering (extract filter does not preserve indices);
    // frontierClusterIndices.clear();
    // kdtree->setInputCloud(frontierCloud);
    // ec.setSearchMethod(kdtree);
    // ec.setInputCloud(frontierCloud);
    // ec.extract(frontierClusterIndices);
    return 1;
  }
}

bool Msfm3d::normalFrontierFilter()
{
  // Create cloud pointer to store the removed points
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr normalCloud(new pcl::PointCloud<pcl::PointXYZINormal>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr frontierCloudPreFilter(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::copyPointCloud(*frontierCloud, *frontierCloudPreFilter);

  // Creating the KdTree object for the search method of the extraction
  pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
  kdtree->setInputCloud(frontierCloud);

  // Initialize euclidean cluster extraction object
  pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::PointXYZINormal> ne;
  ne.setSearchMethod(kdtree);
  ne.setInputCloud(frontierCloud);
  ne.setRadiusSearch(2.1*voxel_size);
  ne.compute(*normalCloud);

  float query[3];
  int idx;
  frontierCloud->clear();
  for (int i=0; i<normalCloud->points.size(); i++) {
    float query[3] = {frontierCloudPreFilter->points[i].x, frontierCloudPreFilter->points[i].y, frontierCloudPreFilter->points[i].z};
    idx = xyz_index3(query);
    if (std::abs(normalCloud->points[i].normal_z) >= normalThresholdZ) {
      frontier[idx] = 0;
    } else {
      frontierCloud->points.push_back(frontierCloudPreFilter->points[i]);
    }
  }

  // Filter frontierCloud to keep only the inliers
  ROS_INFO("Frontier cloud after normal filtering has %d points.", (int)frontierCloud->points.size());

  if ((int)frontierCloud->points.size() < 1) {
    return 0;
  } else {
    return 1;
  }
}

bool filterCloudRadius(const float radius, const pcl::PointXYZ point, const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointIndices::Ptr inliers)
{
  // filterRadius returns the indices (inside inliers->indices) in cloud that are within euclidean distance r of point (x,y,z).

  // Find all indices within r of point (x,y,z)
  float distance_squared;
  float delta_x;
  float delta_y;
  float delta_z;
  float radius_squared = radius*radius;
  for (int i = 0; i<(int)cloud->points.size(); i++) { // This line is vague, consider modifying with a more explicit range of values
    delta_x = point.x - cloud->points[i].x;
    delta_y = point.y - cloud->points[i].y;
    delta_z = point.z - cloud->points[i].z;
    distance_squared = (delta_x*delta_x) + (delta_y*delta_y) + (delta_z*delta_z);
    if (distance_squared < radius_squared) inliers->indices.push_back(i);  // Add indices that are within the radius to inliers
  }

  // Return false if there are no points in cloud within r of point (x,y,z)
  if (inliers->indices.size() > 0)
    return 1;
  else
    return 0;
}

void Msfm3d::greedyGrouping(const float radius, const bool print2File)
{
  // greedGrouping generates a vector of (pcl::PointIndices) where each entry in the vector is a greedily sampled group of points within radius of a randomly sampled frontier member.
  // Algorithm:
  //    0) clusterCount = 0; groupCount = 0;
  //    1) Add all indices in the current cluster (frontierClusterIndices[clusterCount]) to ungrouped
  //    2) While ungroupedIndices.size > 0
  //      a) Sample an index from ungroupedIndices and get its point location (x, y, z)
  //      b) Filter the ungrouped to ranges (x-r,y-r,z-r):(x+r, y+r, z+r) where r is radius
  //      c) Remove all indices that are within r of (x,y,z) from ungroupedIndices and add them to greedyCluster[groupCount]
  //      d) groupCount++;
  //    3) clusterCount++, GoTo 1)
  // Inputs:
  //    - Msfm3d.frontierClusterIndices (std::vector<pcl::PointIndices>)
  //    - Msfm3d.frontierCloud (pcl::PointCloud<pcl::PointXYZ>::Ptr)
  //    - radius (float)
  // Outputs:
  //    - Msfm3d.greedyGroups (std::vector<pcl::PointIndices>)
  //    - Msfm3d.greedyCenters (pcl::PointCloud<pcl::PointXYZ>)
  //
  // Each greedy group is used as a source for the pose sampling function.

  // Intialize random seed:
  std::random_device rd;
  // std::mt19937 mt(19937);
  std::mt19937 mt(rd()); // Really random (much better than using rand())

  // Initialize counts of the current group and cluster in the algorithm
  int groupCount = 0;
  int clusterCount = 0;

  // Clear the greedy cluster arrays in the msfm3d object
  greedyGroups.clear();
  greedyCenters.clear();

  ROS_INFO("Beginning greedy grouping of %d clusters with sizes:", (int)frontierClusterIndices.size());
  for (int i = 0; i < frontierClusterIndices.size(); i++) {
  	std::cout << frontierClusterIndices[i].indices.size() << " ";
  }
  std::cout << std::endl;

  // Loop through all of the frontier clusters
  for (std::vector<pcl::PointIndices>::const_iterator it = frontierClusterIndices.begin(); it != frontierClusterIndices.end(); ++it) {

    // Initialize a PointIndices pointer of all the ungrouped points in the current cluster
    pcl::PointIndices::Ptr ungrouped(new pcl::PointIndices);
    for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit) {
    	ungrouped->indices.push_back(*pit);
    }

    // Go until every member of ungrouped is in a greedy group
    while (ungrouped->indices.size() > 0) {
      // Reset/Update ungrouped PointCloud
      pcl::PointCloud<pcl::PointXYZ>::Ptr ungrouped_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      pcl::ExtractIndices<pcl::PointXYZ> extract;
      extract.setInputCloud(frontierCloud);
      extract.setIndices(ungrouped);
      extract.filter(*ungrouped_cloud);
      // ROS_INFO("Initializing ungrouped_cloud for group %d containing %d points.", groupCount, (int)ungrouped_cloud->points.size());

      // Sample a random index from ungrouped
      std::uniform_int_distribution<int> dist(0, (int)ungrouped->indices.size()-1);
      int sample_id = ungrouped->indices[dist(mt)];
      pcl::PointXYZ sample_point = frontierCloud->points[sample_id];
      // ROS_INFO("Sampled index %d which is located at (%f, %f, %f).", sample_id, sample_point.x, sample_point.y, sample_point.z);

      // Find all the voxels in ungrouped_cloud that are within r of sample_point
      pcl::PointIndices::Ptr group(new pcl::PointIndices);
      filterCloudRadius(radius, sample_point, ungrouped_cloud, group);
      // ROS_INFO("Found %d points within %f of the sample point.", (int)group->indices.size(), radius);

      // Resize greedyGroups to make room for a new group
      greedyGroups.resize(greedyGroups.size() + 1);

      // Add grouped indices to greedyGroups
      for (std::vector<int>::const_iterator git = group->indices.begin(); git != group->indices.end(); ++git) {
        greedyGroups[groupCount].indices.push_back(ungrouped->indices[*git]); // Add indices to group
        ungrouped->indices[*git] = -1;
      }

      // Removed grouped indices from ungrouped  (This command removes all members of ungrouped->indices that have a value of -1)
      ungrouped->indices.erase(std::remove(ungrouped->indices.begin(), ungrouped->indices.end(), -1), ungrouped->indices.end());

      // Add sample_point to greedyCenters
      greedyCenters.points.push_back(sample_point);

      // Add the current cluster number to greedyClusterNumber
      greedyClusterNumber.push_back(clusterCount);

      // Add group clouds to file for debugging
      int j = 0;
      pcl::PCDWriter writer;
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
      for (std::vector<int>::const_iterator fit = greedyGroups[groupCount].indices.begin(); fit != greedyGroups[groupCount].indices.end(); ++fit) {
        if (print2File) cloud_cluster->points.push_back(frontierCloud->points[*fit]);
      }
      if (print2File) {
        cloud_cluster->width = cloud_cluster->points.size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        // std::cout << "PointCloud representing the group: " << cloud_cluster->points.size () << " data points." << std::endl;
        std::stringstream ss;
        ss << "pcl_clusters/cloud_cluster_" << clusterCount << "_group_" << groupCount << ".pcd";
        // std::cout << "Writing to file " << ss.str() << std::endl;
        writer.write<pcl::PointXYZ> (ss.str(), *cloud_cluster, false); //*
      }

      groupCount++;
    }
    clusterCount++;
  }
  ROS_INFO("%d groups generated from the frontier clusters.", groupCount);
}

void Msfm3d::callback_position(const nav_msgs::Odometry msg)
{
  if (!receivedPosition) receivedPosition = 1;
  position[0] = msg.pose.pose.position.x;
  position[1] = msg.pose.pose.position.y;
  position[2] = msg.pose.pose.position.z + 0.4;
  ROS_INFO("Robot pose updated!");
}

void Msfm3d::callback_Octomap(const octomap_msgs::Octomap::ConstPtr msg)
{
  ROS_INFO("Getting OctoMap message...");


  // Free/Allocate the tree memory
  // ROS_INFO("Converting Octomap msg to AbstractOcTree...");
  // octomap::AbstractOcTree* abstree = new octomap::AbstractOcTree(msg->resolution);
  // abstree = octomap_msgs::binaryMsgToMap(*msg); // OcTree object for storing Octomap data.
  // ROS_INFO("Octomap converted to AbstractOcTree.");

  // Check if the message is empty (skip callback if it is)
  if (msg->data.size() == 0) {
    ROS_INFO("Octomap message is of length 0");
    return;
  }

  if (!receivedMap) receivedMap = 1;
  if (!updatedMap) updatedMap = 1;

  delete mytree;
  mytree = new octomap::OcTree(msg->resolution);
  mytree = (octomap::OcTree*)octomap_msgs::binaryMsgToMap(*msg);
  ROS_INFO("AbstractOcTree cast into OcTree.");


  ROS_INFO("Parsing Octomap...");
  // Make sure the tree is at the same resolution as the esdf we're creating.  If it's not, change the resolution.
  ROS_INFO("Tree resolution is %f meters.", mytree->getResolution());
  if ((mytree->getResolution() - (double)voxel_size) > (double)0.01) {
    ROS_INFO("Planner voxel_size is %0.2f and Octomap resolution is %0.2f.  They must be equivalent.", voxel_size, mytree->getResolution());
    receivedMap = 0;
    return;
  }

  // Parse out ESDF struct dimensions from the the AbstractOcTree object
  double x, y, z;
  mytree->getMetricMin(x, y, z);
  ROS_INFO("Got Minimum map dimensions.");
  esdf.min[0] = (float)x - 1.5*voxel_size;
  esdf.min[1] = (float)y - 1.5*voxel_size;
  esdf.min[2] = (float)z - 1.5*voxel_size;
  mytree->getMetricMax(x, y, z);
  ROS_INFO("Got Maximum map dimensions.");
  esdf.max[0] = (float)x + 1.5*voxel_size;
  esdf.max[1] = (float)y + 1.5*voxel_size;
  esdf.max[2] = (float)z + 1.5*voxel_size;
  for (int i=0; i<3; i++) esdf.size[i] = roundf((esdf.max[i]-esdf.min[i])/voxel_size) + 1;

  // Print out the max and min values with the size values.
  ROS_INFO("The (x,y,z) ranges are (%0.2f to %0.2f, %0.2f to %0.2f, %0.2f to %0.2f).", esdf.min[0], esdf.max[0], esdf.min[1], esdf.max[1], esdf.min[2], esdf.max[2]);
  ROS_INFO("The ESDF dimension sizes are %d, %d, and %d.", esdf.size[0], esdf.size[1], esdf.size[2]);

  // Free and allocate memory for the esdf.data and esdf.seen pointer arrays
  delete[] esdf.data;
  esdf.data = NULL;
  esdf.data = new double [esdf.size[0]*esdf.size[1]*esdf.size[2]] { }; // Initialize all values to zero.
  delete[] esdf.seen;
  esdf.seen = NULL;
  esdf.seen = new bool [esdf.size[0]*esdf.size[1]*esdf.size[2]] { }; // Initialize all values to zero.

  // Loop through tree and extract occupancy info into esdf.data and seen/not into esdf.seen
  double size, value;
  float point[3], lower_corner[3];
  int idx, depth, width;
  int lowest_depth = (int)mytree->getTreeDepth();
  int count = 0;
  int freeCount = 0;
  int occCount = 0;
  ROS_INFO("Starting tree iterator on OcTree with max depth %d", lowest_depth);
  for(octomap::OcTree::leaf_iterator it = mytree->begin_leafs(),
       end=mytree->end_leafs(); it!=end; ++it)
  {
    // Get data from node
    depth = (int)it.getDepth();
    point[0] = (float)it.getX();
    point[1] = (float)it.getY();
    point[2] = (float)it.getZ();
    size = it.getSize();
    // if (!(count % 500)) {
    //   ROS_INFO("Binary occupancy at [%0.2f, %0.2f, %0.2f] is: %d", point[0], point[1], point[2], (int)it->getValue());
    //   std::cout << it->getValue() << std::endl;
    // }
    if (it->getValue() > 0) {
      value = 0.0;
      occCount++;
    } else {
      value = 1.0;
      freeCount++;
    }

    // Put data into esdf
    if (depth == lowest_depth){
      idx = xyz_index3(point);
      // std::cout << "Node value: " << it->getValue() << std::endl;
      // ROS_INFO("Assigning an ESDF value at index %d with depth %d and size %f. (%f, %f, %f)", idx, depth, size, point[0], point[1], point[2]);
      esdf.data[idx] = value;
      esdf.seen[idx] = 1;
    } else{ // Fill in all the voxels internal to the leaf
      width = (int)std::pow(2.0, (double)(lowest_depth-depth));
      for (int i=0; i<3; i++){
        lower_corner[i] = point[i] - size/2.0 + voxel_size/2.0;
      }
      // ROS_INFO("Point (%f, %f, %f) is not at the base depth.  It is %d voxels wide.", point[0], point[1], point[2], width);
      // ROS_INFO("Filling in leaf at depth %d with size %f.  The lower corner is at (%f, %f, %f)", depth, size, lower_corner[0], lower_corner[1], lower_corner[2]);
      for (int i=0; i<width; i++){
        point[0] = lower_corner[0] + i*voxel_size;
        for (int j=0; j<width; j++){
          point[1] = lower_corner[1] + j*voxel_size;
          for (int k=0; k<width; k++){
            point[2] = lower_corner[2] + k*voxel_size;
            idx = xyz_index3(point);
            esdf.data[idx] = value;
            esdf.seen[idx] = 1;
          }
        }
      }
    }
  }

  // Initialize the voxels within a vehicle volume around the robot the free.
  if (receivedPosition) {
    float query_point[3];
    int query_idx;
    ROS_INFO("Clearing vehicle volume with limits [%0.2f, %0.2f; %0.2f, %0.2f; %0.2f, %0.2f].", vehicleVolume.xmin,
      vehicleVolume.xmax, vehicleVolume.ymin, vehicleVolume.ymax, vehicleVolume.zmin, vehicleVolume.zmax);
    if (vehicleVolume.set) {
      for(float dx = vehicleVolume.xmin; dx <= vehicleVolume.xmax; dx = dx + voxel_size) {
        query_point[0] = position[0] + dx;
        for(float dy = vehicleVolume.ymin; dy <= vehicleVolume.ymax; dy = dy + voxel_size) {
          query_point[1] = position[1] + dy;
          for(float dz = vehicleVolume.zmin; dz <= vehicleVolume.zmax; dz = dz + voxel_size) {
            query_point[2] = position[2] + dz;
            query_idx = xyz_index3(query_point);
            if ((query_idx >= 0) && (query_idx < esdf.size[0]*esdf.size[1]*esdf.size[2])) { // Check for valid array indices
              esdf.data[query_idx] = 1.0;
              esdf.seen[query_idx] = true;
            }
          }
        }
      }
    }
  }


  ROS_INFO("Octomap message received.  %d leaves labeled as occupied.  %d leaves labeled as free.", occCount, freeCount);
}

void Msfm3d::callback(sensor_msgs::PointCloud2 msg)
{
  ROS_INFO("Getting ESDF PointCloud2...");
  if (msg.data.size() == 0) {
    ROS_INFO("Input PointCloud2 message is empty.");
    return;
  }
  if (!receivedMap) receivedMap = 1;
  if (!updatedMap) updatedMap = 1;
  PC2msg = msg;
  ROS_INFO("ESDF PointCloud2 received!");
}

void Msfm3d::parsePointCloud()
{
  ROS_INFO("Allocating memory for esdf parsing.");
  // integer array to store uint8 byte values
  int bytes[4];
  // local pointcloud storage array
  // float xyzis[5*(PC2msg.width)];
  float * xyzis = new float[5*(PC2msg.width)];
  // pointcloud xyz limits
  float xyzi_max[4], xyzi_min[4];
  int offset;

  // parse pointcloud2 data
  ROS_INFO("Parsing ESDF data into holder arrays.");
  for (int i=0; i<(PC2msg.width); i++) {
    for (int j=0; j<5; j++) {
      if (j>3){ offset = 12; }
      else{ offset = PC2msg.fields[j].offset; }
      for (int k=0; k<4; k++) {
        bytes[k] = PC2msg.data[32*i+offset+k];
      }
      xyzis[5*i+j] = byte2Float(bytes);
      if (j<4){
        if (i < 1){
          xyzi_max[j] = xyzis[5*i+j];
          xyzi_min[j] = xyzis[5*i+j];
        }
        else {
          if (xyzis[5*i+j] > xyzi_max[j]) xyzi_max[j] = xyzis[5*i+j];
          if (xyzis[5*i+j] < xyzi_min[j]) xyzi_min[j] = xyzis[5*i+j];
        }
      }
    }
  }

  // Replace max, min, and size esdf properties
  for (int i=0; i<4; i++) { esdf.max[i] = xyzi_max[i] + voxel_size; esdf.min[i] = xyzi_min[i] - voxel_size; } // Add 1 voxel to the maxes and subtract 1 voxel from the mins to get desired frontier performance
  for (int i=0; i<3; i++) esdf.size[i] = roundf((esdf.max[i]-esdf.min[i])/voxel_size) + 1;

  // Print out the max and min values with the size values.
  ROS_INFO("The (x,y,z) ranges are (%0.2f to %0.2f, %0.2f to %0.2f, %0.2f to %0.2f).", esdf.min[0], esdf.max[0], esdf.min[1], esdf.max[1], esdf.min[2], esdf.max[2]);
  ROS_INFO("The ESDF dimension sizes are %d, %d, and %d.", esdf.size[0], esdf.size[1], esdf.size[2]);

  // Empty current esdf and seen matrix and create a new ones.
  delete[] esdf.data;
  esdf.data = NULL;
  esdf.data = new double [esdf.size[0]*esdf.size[1]*esdf.size[2]] { }; // Initialize all values to zero.
  delete[] esdf.seen;
  esdf.seen = NULL;
  esdf.seen = new bool [esdf.size[0]*esdf.size[1]*esdf.size[2]] { };
  ROS_INFO("ESDF Data is of length %d", esdf.size[0]*esdf.size[1]*esdf.size[2]);
  ROS_INFO("Message width is %d", PC2msg.width);

  // Parse xyzis into esdf_mat
  int index;
  float point[3];
  for (int i=0; i<(5*(PC2msg.width)); i=i+5) {
    point[0] = xyzis[i]; point[1] = xyzis[i+1]; point[2] = xyzis[i+2];
    index = xyz_index3(point);
    if (index > (esdf.size[0]*esdf.size[1]*esdf.size[2])) ROS_INFO("WARNING: Parsing index is greater than array sizes!");
    // Use the hyperbolic tan function from the btraj paper
    // esdf = v_max*(tanh(d - e) + 1)/2
    if (xyzis[i+3] > 0.0) {
      // esdf.data[index] = (double)(5.0*(tanh(xyzis[i+3] - exp(1.0)) + 1.0)/2.0); // doesn't work too well (numerical issues)
      esdf.data[index] = (double)xyzis[i+3];
    }
    else {
      esdf.data[index] = (double)(0.0);
    }
    esdf.seen[index] = (xyzis[i+4]>0.0);
  }

  if (receivedPosition) {
    float query_point[3];
    int query_idx;
    ROS_INFO("Clearing vehicle volume with limits [%0.2f, %0.2f; %0.2f, %0.2f; %0.2f, %0.2f].", vehicleVolume.xmin,
      vehicleVolume.xmax, vehicleVolume.ymin, vehicleVolume.ymax, vehicleVolume.zmin, vehicleVolume.zmax);
    if (vehicleVolume.set) {
      for(float dx = vehicleVolume.xmin; dx <= vehicleVolume.xmax; dx = dx + voxel_size) {
        query_point[0] = position[0] + dx;
        for(float dy = vehicleVolume.ymin; dy <= vehicleVolume.ymax; dy = dy + voxel_size) {
          query_point[1] = position[1] + dy;
          for(float dz = vehicleVolume.zmin; dz <= vehicleVolume.zmax; dz = dz + voxel_size) {
            query_point[2] = position[2] + dz;
            query_idx = xyz_index3(query_point);
            if ((query_idx >= 0) && (query_idx < esdf.size[0]*esdf.size[1]*esdf.size[2])) { // Check for valid array indices
              if (esdf.data[query_idx] < 1.0) esdf.data[query_idx] = 1.0 + inflateWidth;
              esdf.seen[query_idx] = true;
            }
          }
        }
      }
    }
  }

  // Free xyzis holder array memory
  delete[] xyzis;
}

void Msfm3d::index3_xyz(const int index, float point[3])
{
  // x+y*sizx+z*sizx*sizy
  point[2] = esdf.min[2] + (index/(esdf.size[1]*esdf.size[0]))*voxel_size;
  point[1] = esdf.min[1] + ((index % (esdf.size[1]*esdf.size[0]))/esdf.size[0])*voxel_size;
  point[0] = esdf.min[0] + ((index % (esdf.size[1]*esdf.size[0])) % esdf.size[0])*voxel_size;
}

int Msfm3d::xyz_index3(const float point[3])
{
  int ind[3];
  for (int i=0; i<3; i++) ind[i] = roundf((point[i]-esdf.min[i])/voxel_size);
  return mindex3(ind[0], ind[1], ind[2], esdf.size[0], esdf.size[1]);
}

void Msfm3d::updateFrontierMsg()
{
  // Declare a new cloud to store the converted message
  sensor_msgs::PointCloud2 newPointCloud2;

  // Convert from pcl::PointCloud to sensor_msgs::PointCloud2
  pcl::toROSMsg(*frontierCloud, newPointCloud2);
  newPointCloud2.header.seq = 1;
  newPointCloud2.header.stamp = ros::Time();
  newPointCloud2.header.frame_id = frame;

  // Update the old message
  frontiermsg = newPointCloud2;
}

bool updateFrontier(Msfm3d& planner)
{
  ROS_INFO("Beginning Frontier update step...");

  // Expand the frontier vector to be at least as big as the esdf space
  // Make sure to copy the previous frontier data



  //**Plan on updating the frontier only in a sliding window of grid spaces surrounding the vehicle**


  int npixels = planner.esdf.size[0]*planner.esdf.size[1]*planner.esdf.size[2];
  delete[] planner.frontier;
  ROS_INFO("Previous frontier deleted.  Allocating new frontier of size %d...", npixels);
  planner.frontier = NULL;
  planner.frontier = new bool [npixels] { }; // Initialize the frontier array as size npixels with all values false.

  // Initialize the pointCloud version of the frontier
  planner.frontierCloud->clear();
  ROS_INFO("New frontier array added with all values set to false.");

  delete[] planner.entrance;
  planner.entrance = NULL;
  planner.entrance = new bool [npixels] { }; // Initialize the entrance array as size npixels with all values false.

  float point[3], query[3];
  int neighbor[6];
  bool frontier = 0;
  pcl::PointXYZ _point;

  int frontierCount = 0;
  int pass1 = 0;
  int pass2 = 0;
  int pass3 = 0;
  int pass4 = 0;
  int pass5 = 0;

  // Extra variables for ground vehicle case so that only frontier close to vehicle plane are chosen.
  for (int i=0; i<npixels; i++){

    // Check if the voxel has been seen and is unoccupied
    // if (planner.esdf.seen[i] && (planner.esdf.data[i]>0.0) && (dist3(planner.position, point) >= planner.bubble_radius)) {
    if (planner.esdf.seen[i] && (planner.esdf.data[i]>0.0)) {
      pass1++;
      // Get the 3D point location
      planner.index3_xyz(i, point);
      _point.x = point[0];
      _point.y = point[1];
      _point.z = point[2];

      // Check if the voxel is a frontier by querying adjacent voxels
      for (int j=0; j<3; j++) query[j] = point[j];

      // Create an array of neighbor indices
      for (int j=0; j<3; j++){
        // if (point[j] < (planner.esdf.max[j] - planner.voxel_size)) {
          query[j] = point[j] + planner.voxel_size;
        // }
        neighbor[2*j] = planner.xyz_index3(query);
        // if (point[j] > (planner.esdf.min[j] + planner.voxel_size)) {
          query[j] = point[j] - planner.voxel_size;
        // }
        neighbor[2*j+1] = planner.xyz_index3(query);
        query[j] = point[j];
      }

      // Check if the neighbor indices are unseen voxels or on the edge of the map
      if (planner.ground) {
        for (int j=0; j<4; j++) {
          // if (!planner.esdf.seen[neighbor[j]] && !(i == neighbor[j]) && !frontier) {
          bool out_of_bounds = ((neighbor[j] < 0) || neighbor[j] >= npixels);
          if (!planner.esdf.seen[neighbor[j]] || out_of_bounds) {
            frontier = 1;
            pass2++;
            break;
          }
        }
        // Eliminate frontiers with unseen top/bottom neighbors
        // if ((!planner.esdf.seen[neighbor[4]] && i != neighbor[4]) || (!planner.esdf.seen[neighbor[5]] && i != neighbor[5])) {
        //   frontier = 0;
        //   pass3++;
        // }
      }
      else {
        // For the time being, exclude the top/bottom neighbor (last two neighbors)
        for (int j=0; j<6; j++) {
        // for (int j=0; j<4; j++) {
          bool out_of_bounds = ((neighbor[j] < 0) || neighbor[j] >= npixels);
          if (!planner.esdf.seen[neighbor[j]] || out_of_bounds) {
            frontier = 1;
            pass2++;
            break;
          }
        }
        // if (!planner.esdf.seen[neighbor[5]]  && !(i == neighbor[5])) frontier = 1;
      }
      // Check if the point is on the ground if it is a ground robot
      if (frontier) {
        // Only consider frontiers on the floor
        // if (planner.esdf.data[neighbor[5]] > (0.01)) frontier = 0;
        // if (!planner.esdf.seen[neighbor[5]]) frontier = 0;

        // Only consider frontiers close in z-coordinate (temporary hack)
        // if (planner.dzFrontierVoxelWidth > 0) {
        //   if (abs(planner.position[2] - point[2]) >= planner.dzFrontierVoxelWidth*planner.voxel_size) {
        //     pass3++;
        //     frontier = 0;
        //   }
        // }

        // // Eliminate frontiers that are adjacent to occupied cells
        // for (int j=0; j<6; j++) {
        //   if (planner.esdf.data[neighbor[j]] < (0.0217) && planner.esdf.seen[neighbor[j]]) {
        //     pass4++;
        //     frontier = 0;
        //   }
        // }
      }

      // Check if the voxel is at the entrance

      if (frontier && (dist3(point, planner.origin) <= planner.entranceRadius)) {
        pass5++;
        planner.entrance[i] = 1;
        frontier = 0;
      }

      // Check if frontier is in fence
      if (!(planner.inBoundary(point))) {
        frontier = 0;
      }

      // If the current voxel is a frontier, add the  current voxel location to the planner.frontier array
      if (frontier) {
        frontier = 0; // reset for next loop
        planner.frontier[i] = 1;
        frontierCount++;
        planner.frontierCloud->push_back(_point);
      }
    }
  }
  ROS_INFO("%d points are free.  %d points are free and adjacent to unseen voxels.  %d points filtered for being too high/low.  %d points filtered for being adjacent to occupied voxels.  %d points at entrance.", pass1, pass2, pass3, pass4, pass5);
  ROS_INFO("Frontier updated. %d voxels initially labeled as frontier.", frontierCount);

  if (frontierCount == 0) {
    return false;
  }

  // Filter frontiers based upon normal vectors
  if (!planner.normalFrontierFilter()) return false;

  // Cluster the frontier into euclidean distance groups
  if (planner.clusterFrontier(false)) {
    // Group frontier within each cluster with greedy algorithm
    planner.greedyGrouping(4*planner.voxel_size, false);
    return true;
  } else {
    return false;
  }
}


int main(int argc, char **argv)
{
  /**
   * The ros::init() function needs to see argc and argv so that it can perform
   * any ROS arguments and name remapping that were provided at the command line.
   * For programmatic remappings you can use a different version of init() which takes
   * remappings directly, but for most command-line programs, passing argc and argv is
   * the easiest way to do it.  The third argument to init() is the name of the node.
   *
   * You must call one of the versions of ros::init() before using any other
   * part of the ROS system.
   */
  ros::init(argc, argv, "msfm3d");
  /**
   * NodeHandle is the main access point to communications with the ROS system.
   * The first NodeHandle constructed will fully initialize this node, and the last
   * NodeHandle destructed will close down the node.
   */
  ros::NodeHandle n;

  // Initialize planner object
  ROS_INFO("Initializing msfm3d planner...");
  // Voxel size for Octomap or Voxblox
  float voxel_size;
  n.param("global_planning/resolution", voxel_size, (float)0.1);
  Msfm3d planner(voxel_size);

  // Set vehicle type, map type, and global frame name
  bool ground, esdf_or_octomap;
  std::string global_frame;
  n.param("global_planning/isGroundVehicle", ground, false);
  n.param("global_planning/useOctomap", esdf_or_octomap, true); // Use a TSDF/ESDF message PointCloud2 (0) or Use an octomap message (1)
  n.param<std::string>("global_planning/frame_id", global_frame, "world");
  planner.ground = ground;
  if (planner.ground) ROS_INFO("Vehicle type set to ground vehicle.");
  else ROS_INFO("Vehicle type set to air vehicle");
  planner.esdf_or_octomap = esdf_or_octomap;
  planner.frame = global_frame;

  // Clustering Parameters
  float cluster_radius, min_cluster_size;
  n.param("global_planning/cluster_radius", cluster_radius, (float)(1.5*voxel_size)); // voxels
  n.param("global_planning/min_cluster_size", min_cluster_size, (float)(5.0/voxel_size)); // voxels
  n.param("global_planning/normalThresholdZ", planner.normalThresholdZ, (float)1.1);
  planner.cluster_radius = cluster_radius;
  planner.min_cluster_size = min_cluster_size;

  bool fenceOn;
  float fence_x_min, fence_x_max, fence_y_min, fence_y_max, fence_z_min, fence_z_max;
  n.param("global_planning/fenceOn", fenceOn, false);
  n.param("global_planning/fence_xmin", fence_x_min, (float)-50.0);
  n.param("global_planning/fence_xmax", fence_x_max, (float)50.0);
  n.param("global_planning/fence_ymin", fence_y_min, (float)-50.0);
  n.param("global_planning/fence_ymax", fence_y_max, (float)50.0);
  n.param("global_planning/fence_zmin", fence_z_min, (float)-50.0);
  n.param("global_planning/fence_zmax", fence_z_max, (float)50.0);
  planner.bounds.set = fenceOn;
  planner.bounds.xmin = fence_x_min;
  planner.bounds.xmax = fence_x_max;
  planner.bounds.ymin = fence_y_min;
  planner.bounds.ymax = fence_y_max;
  planner.bounds.zmin = fence_z_min;
  planner.bounds.zmax = fence_z_max;

  // Vehicle volume free boundaries
  bool vehicleVolumeOn;
  float vehicleVolumeXmin, vehicleVolumeXmax, vehicleVolumeYmin, vehicleVolumeYmax, vehicleVolumeZmin, vehicleVolumeZmax;
  n.param("global_planning/vehicleVolumeOn", vehicleVolumeOn, false);
  n.param("global_planning/vehicleVolumeXmin", vehicleVolumeXmin, (float)-0.50);
  n.param("global_planning/vehicleVolumeXmax", vehicleVolumeXmax, (float)0.50);
  n.param("global_planning/vehicleVolumeYmin", vehicleVolumeYmin, (float)-0.50);
  n.param("global_planning/vehicleVolumeYmax", vehicleVolumeYmax, (float)0.50);
  n.param("global_planning/vehicleVolumeZmin", vehicleVolumeZmin, (float)-0.50);
  n.param("global_planning/vehicleVolumeZmax", vehicleVolumeZmax, (float)0.50);
  planner.vehicleVolume.set = vehicleVolumeOn;
  planner.vehicleVolume.xmin = vehicleVolumeXmin;
  planner.vehicleVolume.xmax = vehicleVolumeXmax;
  planner.vehicleVolume.ymin = vehicleVolumeYmin;
  planner.vehicleVolume.ymax = vehicleVolumeYmax;
  planner.vehicleVolume.zmin = vehicleVolumeZmin;
  planner.vehicleVolume.zmax = vehicleVolumeZmax;

  // Get planner operating rate in Hz
  float updateRate;
  n.param("global_planning/updateRate", updateRate, (float)1.0); // Hz

  ROS_INFO("Subscribing to Occupancy Grid...");
  ros::Subscriber sub0 = n.subscribe("octomap_binary", 1, &Msfm3d::callback_Octomap, &planner);

  ROS_INFO("Subscribing to ESDF or TSDF PointCloud2...");
  ros::Subscriber sub1 = n.subscribe("voxblox_node/tsdf_pointcloud", 1, &Msfm3d::callback, &planner);

  ROS_INFO("Subscribing to robot state...");
  ros::Subscriber sub2 = n.subscribe("odometry", 1, &Msfm3d::callback_position, &planner);

  ros::Publisher pub1 = n.advertise<geometry_msgs::PointStamped>("nearest_frontier", 5);

  ros::Publisher pub3 = n.advertise<sensor_msgs::PointCloud2>("frontier", 5);



  int i = 0;
  ros::Rate r(updateRate); // Hz
  clock_t tStart;
  int npixels;
  int spins = 0;
  int oldFrontierClusterCount = 0;


  ROS_INFO("Starting planner...");
  r.sleep();
  while (ros::ok())
  {
    r.sleep();
    ros::spinOnce();
    ROS_INFO("Planner Okay.");
    if (planner.receivedMap && !planner.esdf_or_octomap){
      planner.parsePointCloud();
    }
    // Heartbeat status update
    if (planner.receivedPosition) {
      ROS_INFO("Position: [x: %f, y: %f, z: %f]", planner.position[0], planner.position[1], planner.position[2]);
      i = planner.xyz_index3(planner.position);
      ROS_INFO("Index at Position: %d", i);
      if (planner.receivedMap) {
        ROS_INFO("ESDF or Occupancy at Position: %f", planner.esdf.data[i]);
        // Find frontier cells and add them to planner.frontier
        if (updateFrontier(planner)) {
          planner.updateFrontierMsg();
          pub3.publish(planner.frontiermsg);
          ROS_INFO("Frontier published!");
        }
      }
    }

  }

  return 0;
}
