#include "image_converter.hpp"

static const std::string OPENCV_WINDOW = "Image window";

int cv_index = 0;

ImageConverter::ImageConverter() : it_(nh_)
{
  // person_tracker = PersonTracker();

  // Subscribe to input video feed and publish output video feed
  image_sub_ = it_.subscribe("/D435i/image_raw", 100, &ImageConverter::imageCb, this);
  image_pub_ = it_.advertise("/HSE/frame_with_bbox", 1);
  // cv::namedWindow(OPENCV_WINDOW);
  test = "test";
}

ImageConverter::~ImageConverter()
{
}

void ImageConverter::imageCb(const sensor_msgs::ImageConstPtr &msg)
{
  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(msg);
  }
  catch (cv_bridge::Exception &e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  cv_index += 1;
  // Send Image to Paul's code
  cv::Mat labeled_img;
  labeled_img = person_tracker.Run(cv_ptr->image, cv_index);
  
  // Get frame back and convert back to sensor image
  cv_bridge::CvImage img_bridge;
  sensor_msgs::Image img_msg;
  std_msgs::Header header; // empty header
  header.seq = cv_index; 
  header.stamp = ros::Time::now(); // time
  img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, labeled_img);
  img_bridge.toImageMsg(img_msg);

  // Output modified video stream
  image_pub_.publish(img_msg);
}
