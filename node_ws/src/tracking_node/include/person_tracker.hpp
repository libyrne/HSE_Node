#ifndef PERSON_TRACKER_H
#define PERSON_TRACKER_H

#include <stddef.h>
#include <stdint.h>
#include <iostream>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gflags/gflags.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>

#include <models/detection_model.h>
#include <models/detection_model_centernet.h>
#include <models/detection_model_ssd.h>
#include <models/detection_model_yolo.h>
#include <models/input_data.h>
#include <models/internal_model_data.h>
#include <models/model_base.h>
#include <models/results.h>
#include <monitors/presenter.h>
#include <pipelines/metadata.h>
#include <utils/common.hpp>
#include <utils/config_factory.h>
#include <utils/images_capture.h>
#include <utils/ocv_common.hpp>
#include <utils/performance_metrics.hpp>
#include <utils/slog.hpp>

#include "cnn.hpp"
#include "core.hpp"
#include "descriptor.hpp"
#include "distance.hpp"
// #include "pedestrian_tracker_demo.hpp"
#include "tracker.hpp"
#include "utils.hpp"


struct Initialize
{

    std::string i = "/home/swoopdaddywhoop/hse_ws/src/Human-State-Estimation/multi-angle-near-wall_pE8qpWQR.mp4"; // input directory
    std::string det_model = "/home/swoopdaddywhoop/hse_ws/src/Human-State-Estimation/open_model_zoo/tools/model_tools/intel/person-detection-retail-0013/FP32/person-detection-retail-0013.xml";
    std::string reid_model = "/home/swoopdaddywhoop/hse_ws/src/Human-State-Estimation/open_model_zoo/tools/model_tools/intel/person-reidentification-retail-0277/FP32/person-reidentification-retail-0277.xml";
    std::string detlog_out = "/home/swoopdaddywhoop/hse_ws/src/Human-State-Estimation/output_NEU.mp4";
    std::string at = "ssd";

    bool h = false;

    uint32_t first = 0;
    uint32_t read_limit = static_cast<gflags::uint32>(std::numeric_limits<size_t>::max());

    bool loop = false;

    std ::string o = "/home/swoopdaddywhoop/hse_ws/src/Human-State-Estimation/output_NEU.mp4"; // change it depending on your
    uint32_t limit = 1000;
    std::string d_det = "CPU";
    std::string d_reid = "CPU";
    std::string layout_det = "";
    bool r = false;
    bool no_show = false;
    int32_t delay = 3;
    std::string u = "";
    double t = 0.5;
    bool auto_resize = false;
    double iou_t = 0.5;
    bool yolo_af = true;

    uint32_t nireq = 0;
    uint32_t nthreads = 0;
    std::string nstreams = "";
    int32_t person_label = -1;

    std::string detector_mode = d_det;
    std::string reid_mode = d_reid;

    bool should_print_out = r;

    bool should_show = !no_show;

    bool should_save_det_log = !detlog_out.empty();
};

class PersonTracker
{
private:
    /* data */
public:
    PersonTracker(/* args */);
    ~PersonTracker();
    cv::Mat TrackPerson(cv::Mat frame,
                    std::unique_ptr<ModelBase> &detectionModel,
                    ov::InferRequest req,
                    int32_t person_label,
                    std::unique_ptr<PedestrianTracker> &tracker,
                    double video_fps,
                    unsigned frameIdx);
    cv::Mat Run(cv::Mat frame, unsigned frameIdx);
    std::unique_ptr<PedestrianTracker> CreatePedestrianTracker(const std::string &reid_model,
                                                           const ov::Core &core,
                                                           const std::string &deviceName,
                                                           bool should_keep_tracking_info);
};

#endif