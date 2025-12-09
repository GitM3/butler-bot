// RF-DETR C++ ROS2 node
// Mirrors ROS/src/py_detr/py_detr/rf_detr_node.py behavior

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/string.hpp>

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>

#include <librealsense2/rs.hpp>

#include <onnxruntime_cxx_api.h>
#ifdef ENABLE_ORT_CUDA
#include <onnxruntime_c_api.h>
#endif

using std::placeholders::_1;
using namespace std::chrono_literals;

namespace {
// Colors (B,G,R)
const cv::Scalar COL_WHITE(255, 255, 255);
const cv::Scalar COL_TARGET(0, 200, 255);
const cv::Scalar COL_CONTOUR(0, 255, 0);
const cv::Scalar COL_SEARCH(255, 165, 0);

// COCO subset used by the Python node
static const std::vector<std::pair<int, std::string>> COCO_CLASSES = {
    {44, "bottle"},
    {46, "wine glass"},
    {47, "cup"},
};

inline float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }

inline void box_cxcywh_to_xyxy(cv::Mat &boxes) {
  // boxes: Nx4 (cx,cy,w,h) in-place convert to (x1,y1,x2,y2)
  for (int i = 0; i < boxes.rows; ++i) {
    float cx = boxes.at<float>(i, 0);
    float cy = boxes.at<float>(i, 1);
    float w = boxes.at<float>(i, 2);
    float h = boxes.at<float>(i, 3);
    float x1 = cx - w * 0.5f;
    float y1 = cy - h * 0.5f;
    float x2 = cx + w * 0.5f;
    float y2 = cy + h * 0.5f;
    boxes.at<float>(i, 0) = x1;
    boxes.at<float>(i, 1) = y1;
    boxes.at<float>(i, 2) = x2;
    boxes.at<float>(i, 3) = y2;
  }
}

enum class State { DETECT = 0, SEARCH = 1, TRACK = 2, FINISH = 3 };
static const char *STATES[] = {"DETECT", "SEARCH", "TRACK", "FINISH"};

struct Detection {
  int class_id;
  std::string class_name;
  float score;
  cv::Rect2f bbox; // absolute pixels
  cv::Point2f center;
  std::optional<float> depth; // in meters
};

class RFDETR_ONNX {
public:
  RFDETR_ONNX(const std::string &model_path, bool use_cuda)
      : env_(ORT_LOGGING_LEVEL_WARNING, "rfdetr"), session_options_(),
        allocator_(), use_cuda_(use_cuda) {
    // Default to CPU EP; optionally append CUDA if available (see below)
    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetInterOpNumThreads(1);

    // Optionally enable CUDA EP (compiled only when provider libs are linked)
#ifdef ENABLE_ORT_CUDA
    if (use_cuda_) {
      try {
#if ORT_API_VERSION >= 12
        OrtCUDAProviderOptionsV2 *cuda_options = nullptr;
        auto &api = Ort::GetApi();
        Ort::ThrowOnError(api.CreateCUDAProviderOptions(&cuda_options));
        // Use defaults; could set keys/values here if needed
        Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_CUDA_V2(
            session_options_, cuda_options));
        Ort::ThrowOnError(api.ReleaseCUDAProviderOptions(cuda_options));
#else
        // Legacy API fallback
        Ort::ThrowOnError(
            OrtSessionOptionsAppendExecutionProvider_CUDA(session_options_, 0));
#endif
      } catch (const Ort::Exception &e) {
        // Fall back to CPU if CUDA EP append fails
        fprintf(stderr,
                "[rfdetr_cpp] CUDA EP unavailable; falling back to CPU: %s\n",
                e.what());
      }
    }
#endif

    try {
      session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(),
                                                session_options_);
    } catch (const Ort::Exception &e) {
      throw std::runtime_error(std::string("Failed to load ONNX model: ") +
                               e.what());
    }

    // Input info
    Ort::TypeInfo tinfo = session_->GetInputTypeInfo(0);
    auto tensor_info = tinfo.GetTensorTypeAndShapeInfo();
    auto shape = tensor_info.GetShape(); // NCHW
    {
      auto name_alloc = session_->GetInputNameAllocated(0, allocator_);
      input_name_ = name_alloc.get();
    }

    fixed_h_ =
        (shape.size() >= 3 && shape[2] > 0) ? static_cast<int>(shape[2]) : -1;
    fixed_w_ =
        (shape.size() >= 4 && shape[3] > 0) ? static_cast<int>(shape[3]) : -1;
  }

  // Returns scores, labels, boxes_xyxy_abs (pixels)
  void predict(const cv::Mat &rgb_or_bgr, std::vector<float> &scores,
               std::vector<int> &labels, cv::Mat &boxes_xyxy_abs,
               float conf_thresh = 0.4f, int max_boxes = 50,
               const std::vector<int> *allowed_cids = nullptr) {

    CV_Assert(rgb_or_bgr.type() == CV_8UC3 || rgb_or_bgr.type() == CV_8UC4);
    const int origin_h = rgb_or_bgr.rows;
    const int origin_w = rgb_or_bgr.cols;

    // Preprocess: drop alpha, resize, normalize (ImageNet)
    cv::Mat img = rgb_or_bgr;
    if (img.channels() == 4) {
      cv::Mat tmp;
      cv::cvtColor(img, tmp, cv::COLOR_BGRA2BGR);
      img = tmp;
    }
    int h_in = fixed_h_ > 0 ? fixed_h_ : img.rows;
    int w_in = fixed_w_ > 0 ? fixed_w_ : img.cols;
    if (img.rows != h_in || img.cols != w_in)
      cv::resize(img, img, cv::Size(w_in, h_in), 0, 0, cv::INTER_LINEAR);

    // NOTE: Python passed BGR straight into normalization; mirror that here.
    img.convertTo(img, CV_32F, 1.0 / 255.0);
    cv::Mat chan[3];
    cv::split(img, chan);
    // ImageNet means/stds
    const float MEANS[3] = {0.485f, 0.456f, 0.406f};
    const float STDS[3] = {0.229f, 0.224f, 0.225f};
    for (int c = 0; c < 3; ++c) {
      chan[c] = (chan[c] - MEANS[c]) / STDS[c];
    }
    cv::merge(chan, 3, img);

    // HWC -> CHW, N=1
    std::vector<float> input_tensor_values(1 * 3 * h_in * w_in);
    size_t idx = 0;
    for (int c = 0; c < 3; ++c) {
      for (int y = 0; y < h_in; ++y) {
        for (int x = 0; x < w_in; ++x) {
          input_tensor_values[idx++] = img.at<cv::Vec3f>(y, x)[c];
        }
      }
    }

    std::array<int64_t, 4> input_shape{1, 3, h_in, w_in};
    Ort::MemoryInfo mem_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, input_tensor_values.data(), input_tensor_values.size(),
        input_shape.data(), input_shape.size());

    const char *input_names[] = {input_name_.c_str()};
    // Output: assume first two are boxes (N,Q,4) and logits (N,Q,C)
    std::vector<const char *> output_names;
    size_t out_count = session_->GetOutputCount();
    output_names.reserve(out_count);
    std::vector<Ort::AllocatedStringPtr> out_name_holders;
    out_name_holders.reserve(out_count);
    for (size_t i = 0; i < out_count; ++i) {
      out_name_holders.emplace_back(
          session_->GetOutputNameAllocated(i, allocator_));
      output_names.push_back(out_name_holders.back().get());
    }

    auto outputs =
        session_->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1,
                      output_names.data(), output_names.size());

    // boxes -> (Q,4) normalized cxcywh
    // logits -> (Q,C)
    Ort::Value &boxes_val = outputs.at(0);
    Ort::Value &logits_val = outputs.at(1);

    auto boxes_info = boxes_val.GetTensorTypeAndShapeInfo();
    auto logits_info = logits_val.GetTensorTypeAndShapeInfo();

    std::vector<int64_t> bshape = boxes_info.GetShape();
    std::vector<int64_t> lshape = logits_info.GetShape();
    int64_t Q = bshape.size() >= 2 ? bshape[1] : 0; // N,Q,4
    int64_t C = lshape.size() >= 2 ? lshape[2] : 0; // N,Q,C

    const float *boxes_ptr = boxes_val.GetTensorData<float>();
    const float *logits_ptr = logits_val.GetTensorData<float>();

    // Convert to cv::Mat for convenience
    cv::Mat boxes(static_cast<int>(Q), 4, CV_32F);
    cv::Mat logits(static_cast<int>(Q), static_cast<int>(C), CV_32F);
    std::memcpy(boxes.data, boxes_ptr, sizeof(float) * Q * 4);
    std::memcpy(logits.data, logits_ptr, sizeof(float) * Q * C);

    // probs = sigmoid(logits)
    cv::Mat probs = logits.clone();
    for (int i = 0; i < probs.rows; ++i) {
      for (int j = 0; j < probs.cols; ++j) {
        probs.at<float>(i, j) = sigmoid(probs.at<float>(i, j));
      }
    }

    // scores/labels = max over classes
    scores.clear();
    labels.clear();
    scores.reserve(probs.rows);
    labels.reserve(probs.rows);
    for (int i = 0; i < probs.rows; ++i) {
      double minv, maxv;
      int minl, maxl;
      cv::minMaxIdx(probs.row(i), &minv, &maxv, &minl, &maxl);
      scores.push_back(static_cast<float>(maxv));
      labels.push_back(maxl);
    }

    // top-K by score
    std::vector<int> order(scores.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return scores[a] > scores[b]; });
    if ((int)order.size() > max_boxes)
      order.resize(max_boxes);

    std::vector<float> scores_f;
    scores_f.reserve(order.size());
    std::vector<int> labels_f;
    labels_f.reserve(order.size());
    cv::Mat boxes_f(order.size(), 4, CV_32F);
    int keep_idx = 0;
    for (int idx : order) {
      if (scores[idx] <= conf_thresh)
        continue;
      if (allowed_cids) {
        bool ok = false;
        for (int cid : *allowed_cids)
          if (cid == labels[idx]) {
            ok = true;
            break;
          }
        if (!ok)
          continue;
      }
      scores_f.push_back(scores[idx]);
      labels_f.push_back(labels[idx]);
      boxes.row(idx).copyTo(boxes_f.row(keep_idx++));
    }
    boxes_f = boxes_f.rowRange(0, keep_idx);

    // Convert to xyxy and scale to image size
    box_cxcywh_to_xyxy(boxes_f);
    for (int i = 0; i < boxes_f.rows; ++i) {
      boxes_f.at<float>(i, 0) *= static_cast<float>(origin_w);
      boxes_f.at<float>(i, 2) *= static_cast<float>(origin_w);
      boxes_f.at<float>(i, 1) *= static_cast<float>(origin_h);
      boxes_f.at<float>(i, 3) *= static_cast<float>(origin_h);
    }

    scores.swap(scores_f);
    labels.swap(labels_f);
    boxes_xyxy_abs = boxes_f;
  }

  int fixed_h() const { return fixed_h_; }
  int fixed_w() const { return fixed_w_; }

private:
  Ort::Env env_;
  Ort::SessionOptions session_options_;
  Ort::AllocatorWithDefaultOptions allocator_;
  std::unique_ptr<Ort::Session> session_;
  std::string input_name_;
  int fixed_h_ = -1;
  int fixed_w_ = -1;
  bool use_cuda_ = true; // attempt CUDA if compiled with provider support
};

class RFDetrNode : public rclcpp::Node {
public:
  RFDetrNode() : Node("rfdetr_node_cpp") {
    // Parameters
    model_path_ = declare_parameter<std::string>("model_path", "base.onnx");
    target_class_name_ =
        declare_parameter<std::string>("target_class", "bottle");
    depth_scale_ = declare_parameter<double>("depth_scale", 1.0);

    finish_time_ = declare_parameter<double>("finish_time", 15.0);
    depth_mask_threshold_mm_ =
        declare_parameter<double>("depth_mask_threshold_mm", 700.0);
    depth_mask_margin_mm_ =
        declare_parameter<double>("depth_mask_margin_mm", 20.0);
    depth_mask_percentile_ =
        declare_parameter<double>("depth_mask_percentile", 2.0);
    depth_kernel_size_ = declare_parameter<int>("depth_kernel_size", 21);
    depth_min_contour_area_ =
        declare_parameter<double>("depth_min_contour_area", 500.0);
    track_window_px_ = declare_parameter<double>("track_window_px", 120.0);
    search_bbox_margin_ = declare_parameter<double>("search_bbox_margin", 0.1);
    depth_threshold_mm_ = declare_parameter<double>("depth_threshold", 600.0);
    bag_path_ = declare_parameter<std::string>("bag_path", "");
    frame_rate_ = declare_parameter<double>("frame_rate", 20.0);

    if (depth_kernel_size_ < 3)
      depth_kernel_size_ = 3;
    if (depth_kernel_size_ % 2 == 0)
      depth_kernel_size_ += 1;
    if (track_window_px_ < 20)
      track_window_px_ = 20;

    // Resolve target labels
    target_labels_.clear();
    for (const auto &kv : COCO_CLASSES) {
      if (kv.second == "cup" || kv.second == "bottle" ||
          kv.second == "wine glass")
        target_labels_.push_back(kv.first);
    }
    RCLCPP_INFO(get_logger(),
                "Target labels: [%d,%d,%d] (['cup','bottle','wine glass'])",
                target_labels_[0], target_labels_[1], target_labels_[2]);

    target_class_id_ = -1;
    for (const auto &kv : COCO_CLASSES) {
      if (kv.second == target_class_name_)
        target_class_id_ = kv.first;
    }
    if (target_class_id_ < 0) {
      throw std::runtime_error("Unknown target_class in COCO subset: " +
                               target_class_name_);
    }
    RCLCPP_INFO(get_logger(), "Tracking TARGET_CLASS='%s' (id=%d)",
                target_class_name_.c_str(), target_class_id_);

    // Publishers
    image_pub_ =
        create_publisher<sensor_msgs::msg::Image>("/camera/annotated", 10);
    target_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(
        "/target/center", 10);
    servo_pub_ = create_publisher<std_msgs::msg::Float64>("/set_position", 10);
    yaw_pub_ = create_publisher<std_msgs::msg::Float64>("/set_yaw", 10);
    state_pub_ = create_publisher<std_msgs::msg::String>("/detector/state", 10);

    // ONNX model (attempt CUDA if available and permitted)
    use_cuda_param_ = declare_parameter<bool>("use_cuda", true);
    model_ = std::make_unique<RFDETR_ONNX>(model_path_, use_cuda_param_);

    // State
    state_ = State::DETECT;
    prev_state_ = State::FINISH;

    prev_t_ = now_sec();

    // Control params
    pitch_min_ = 0.0;
    pitch_max_ = 90.0;
    pitch_angle_ = 30.0;
    pitch_angle_prev_ = 0.0;
    yaw_angle_ = 0.0;
    yaw_angle_prev_ = 0.0;

    image_h_ = 480;
    image_w_ = 640;
    image_cx_ = image_w_ / 2.0;
    image_cy_ = image_h_ / 2.0;
    x_max_ = static_cast<int>(0.2 * image_w_);
    y_max_ = static_cast<int>(0.2 * image_h_);

    // RealSense
    if (!bag_path_.empty()) {
      config_.enable_device_from_file(bag_path_);
    } else {
      config_.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
      config_.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    }
    align_ = std::make_unique<rs2::align>(RS2_STREAM_COLOR);
    pipe_.start(config_);
    if (!bag_path_.empty()) {
      rs2::device dev = pipe_.get_active_profile().get_device();
      rs2::playback pb = dev.as<rs2::playback>();
      if (pb)
        pb.set_real_time(false);
    }

    // Post-processing filters
    spatial_filter_.set_option(RS2_OPTION_FILTER_MAGNITUDE, 1);
    spatial_filter_.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, 0.5f);
    spatial_filter_.set_option(RS2_OPTION_FILTER_SMOOTH_DELTA, 10.f);
    spatial_filter_.set_option(RS2_OPTION_HOLES_FILL, 0.f);
    temporal_filter_.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, 0.4f);
    temporal_filter_.set_option(RS2_OPTION_FILTER_SMOOTH_DELTA, 20.f);

    timer_ = create_wall_timer(std::chrono::duration<double>(1.0 / frame_rate_),
                               std::bind(&RFDetrNode::on_timer, this));

    RCLCPP_INFO(get_logger(), "✅ BB Detector C++ ready");
  }

private:
  void on_timer() {
    // Acquire frames
    rs2::frameset frames = pipe_.wait_for_frames();
    frames = align_->process(frames);
    rs2::depth_frame depth_frame = frames.get_depth_frame();
    rs2::video_frame color_frame = frames.get_color_frame();
    if (!depth_frame || !color_frame) {
      RCLCPP_WARN(get_logger(), "No frames received!");
      return;
    }

    // Post-process depth
    depth_frame = post_process_depth_frame(depth_frame);

    // Wrap to OpenCV
    cv::Mat depth_img(
        cv::Size(depth_frame.get_width(), depth_frame.get_height()), CV_16UC1,
        (void *)depth_frame.get_data(), cv::Mat::AUTO_STEP);
    cv::Mat frame(cv::Size(color_frame.get_width(), color_frame.get_height()),
                  CV_8UC3, (void *)color_frame.get_data(), cv::Mat::AUTO_STEP);
    cv::Mat annotated = frame.clone();

    image_h_ = frame.rows;
    image_w_ = frame.cols;
    image_cx_ = image_w_ / 2.0;
    image_cy_ = image_h_ / 2.0;
    x_max_ = static_cast<int>(0.5 * image_w_);
    y_max_ = static_cast<int>(0.5 * image_h_);

    // Run detection
    std::vector<float> scores;
    std::vector<int> labels;
    cv::Mat boxes;
    model_->predict(frame, scores, labels, boxes, 0.4f, 50, &target_labels_);

    // Build detections list and annotate boxes
    std::vector<Detection> detections;
    std::optional<cv::Point2f> det_center;
    std::optional<cv::Rect2f> det_bbox;
    for (int i = 0; i < boxes.rows; ++i) {
      cv::Rect2f r(boxes.at<float>(i, 0), boxes.at<float>(i, 1),
                   boxes.at<float>(i, 2) - boxes.at<float>(i, 0),
                   boxes.at<float>(i, 3) - boxes.at<float>(i, 1));
      float cx = (r.x + r.x + r.width) * 0.5f;
      float cy = (r.y + r.y + r.height) * 0.5f;
      auto depth = get_depth_at(depth_img, cx, cy);
      int cid = labels[i];
      std::string cname = class_name_from_id(cid);

      detections.push_back(
          Detection{cid, cname, scores[i], r, cv::Point2f(cx, cy), depth});

      // Draw all detections
      cv::rectangle(annotated, r, COL_TARGET, 2);
      char buf[64];
      snprintf(buf, sizeof(buf), "%s %.2f", cname.c_str(), scores[i]);
      cv::putText(annotated, buf, cv::Point(r.x, std::max(0.f, r.y - 5)),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, COL_WHITE, 2);
    }

    // Choose target detection = highest score for target_class_id_
    int best_idx = -1;
    float best_score = -1.0f;
    for (size_t i = 0; i < detections.size(); ++i) {
      if (detections[i].class_id == target_class_id_) {
        if (detections[i].score > best_score) {
          best_score = detections[i].score;
          best_idx = (int)i;
        }
      }
    }
    if (best_idx >= 0) {
      det_center = detections[best_idx].center;
      det_bbox = detections[best_idx].bbox;
    }

    // State machine inputs
    bool object_detected_raw =
        (best_idx >= 0) && detections[best_idx].depth.has_value() &&
        (detections[best_idx].depth.value() * 1000.0 <= depth_threshold_mm_);

    if (object_detected_raw) {
      detect_yes_++;
      detect_no_ = 0;
    } else {
      detect_no_++;
      detect_yes_ = 0;
    }

    bool object_detected_stable = (detect_yes_ >= DETECT_YES_THRESH_);
    bool object_lost_stable = (detect_no_ >= DETECT_NO_THRESH_);

    // Depth mask and contour
    std::optional<cv::Rect> mask_bbox_draw;
    std::optional<cv::Point> contour_center;
    std::optional<std::vector<cv::Point>> contour_opt;

    if (state_ == State::DETECT) {
      if (object_detected_stable) {
        RCLCPP_INFO(get_logger(), "STATE → SEARCH (object stable)");
        state_ = State::SEARCH;
        prev_contour_center_.reset();
        contour_yes_ = contour_no_ = 0;
      }
    } else if (state_ == State::SEARCH) {
      cv::Rect track_bbox;
      if (det_bbox) {
        track_bbox =
            expand_bbox(*det_bbox, search_bbox_margin_, depth_img.size());
        mask_bbox_draw = track_bbox;
      } else if (prev_contour_center_) {
        track_bbox = bbox_from_center(*prev_contour_center_,
                                      (int)track_window_px_, depth_img.size());
        mask_bbox_draw = track_bbox;
      } else {
        track_bbox = bbox_from_center(cv::Point((int)image_cx_, (int)image_cy_),
                                      (int)track_window_px_, depth_img.size());
        mask_bbox_draw = track_bbox;
      }

      cv::Mat depth_mask = build_depth_mask(depth_img, track_bbox);
      auto contour_info = find_depth_contour(depth_mask, track_bbox);
      if (contour_info) {
        prev_contour_center_ = contour_info->first;
        contour_center = contour_info->first;
        contour_opt = contour_info->second;
        contour_yes_++;
        contour_no_ = 0;
      } else {
        contour_yes_ = 0;
        contour_no_++;
      }

      bool contour_found_stable = (contour_yes_ >= CONTOUR_YES_THRESH_);
      if (object_lost_stable && !contour_found_stable) {
        RCLCPP_INFO(get_logger(),
                    "STATE → DETECT (lost before confirming contour)");
        state_ = State::DETECT;
        prev_contour_center_.reset();
        contour_yes_ = contour_no_ = 0;
      } else if (contour_found_stable && object_lost_stable) {
        RCLCPP_INFO(get_logger(), "STATE → TRACK (contour confirmed)");
        state_ = State::TRACK;
        contour_no_ = 0;
        track_stable_start_.reset();
      }
    } else if (state_ == State::TRACK) {
      if (object_detected_stable) {
        RCLCPP_INFO(get_logger(), "STATE → DETECT (object back in view)");
        state_ = State::DETECT;
        prev_contour_center_.reset();
        contour_yes_ = contour_no_ = 0;
        track_stable_start_.reset();
      } else {
        cv::Rect track_bbox;
        if (det_bbox)
          track_bbox =
              expand_bbox(*det_bbox, search_bbox_margin_, depth_img.size());
        else if (prev_contour_center_)
          track_bbox = bbox_from_center(
              *prev_contour_center_, (int)track_window_px_, depth_img.size());
        else
          track_bbox =
              bbox_from_center(cv::Point((int)image_cx_, (int)image_cy_),
                               (int)track_window_px_, depth_img.size());
        mask_bbox_draw = track_bbox;

        cv::Mat depth_mask = build_depth_mask(depth_img, track_bbox);
        auto contour_info = find_depth_contour(depth_mask, track_bbox);
        if (contour_info) {
          prev_contour_center_ = contour_info->first;
          contour_center = contour_info->first;
          contour_opt = contour_info->second;
          contour_yes_++;
          contour_no_ = 0;
          if (!track_stable_start_)
            track_stable_start_ = now_sec();
        } else {
          contour_no_++;
          contour_yes_ = 0;
          track_stable_start_.reset();
        }

        bool contour_lost_stable = (contour_no_ >= CONTOUR_NO_THRESH_);
        if (contour_lost_stable) {
          RCLCPP_INFO(get_logger(), "STATE → DETECT (contour lost)");
          state_ = State::DETECT;
          prev_contour_center_.reset();
          contour_yes_ = contour_no_ = 0;
          track_stable_start_.reset();
        } else if (contour_info && track_stable_start_) {
          if (now_sec() - *track_stable_start_ >= finish_time_) {
            RCLCPP_INFO(get_logger(), "STATE → FINISH (contour stable > %.1f)",
                        finish_time_);
            state_ = State::FINISH;
            track_stable_start_.reset();
            contour_yes_ = contour_no_ = 0;
          }
        }
      }
    } else if (state_ == State::FINISH) {
      if (object_detected_stable) {
        RCLCPP_INFO(get_logger(),
                    "STATE → DETECT (object detected during FINISH)");
        state_ = State::DETECT;
        prev_contour_center_.reset();
        contour_yes_ = contour_no_ = 0;
        track_stable_start_.reset();
      }
    }

    // Draw ROI box used for depth mask
    if (mask_bbox_draw) {
      cv::rectangle(annotated, *mask_bbox_draw, COL_SEARCH, 2);
    }

    // Servo control
    std::optional<cv::Point2f> target_point;
    if (contour_opt && contour_center) {
      std::vector<std::vector<cv::Point>> contours = {*contour_opt};
      cv::drawContours(annotated, contours, -1, COL_CONTOUR, 2);
      cv::circle(annotated, *contour_center, 4, COL_CONTOUR, -1);
      target_point =
          cv::Point2f((float)contour_center->x, (float)contour_center->y);
    } else if (det_center) {
      target_point = *det_center;
    }

    if (target_point) {
      float cx = target_point->x;
      float cy = target_point->y;
      float y_err = (float)image_cy_ - cy;
      float x_err = (float)image_cx_ - cx;
      if (std::abs(y_err) > y_max_) {
        pitch_angle_ += 5.0 * (y_err) / (float)image_h_;
        pitch_angle_ = std::clamp<double>(pitch_angle_, pitch_min_, pitch_max_);
      }
      if (std::abs(x_err) > x_max_) {
        yaw_angle_ += 5.0 * (x_err) / (float)image_w_;
      }
    }

    if (pitch_angle_ != pitch_angle_prev_) {
      pitch_angle_prev_ = pitch_angle_;
      std_msgs::msg::Float64 m;
      m.data = pitch_angle_;
      servo_pub_->publish(m);
    }
    if (yaw_angle_ != yaw_angle_prev_) {
      yaw_angle_prev_ = yaw_angle_;
      std_msgs::msg::Float64 m;
      m.data = yaw_angle_;
      yaw_pub_->publish(m);
    }

    if (state_ != prev_state_) {
      std_msgs::msg::String m;
      m.data = STATES[(int)state_];
      state_pub_->publish(m);
      prev_state_ = state_;
    }

    double now = now_sec();
    double fps = 1.0 / (now - prev_t_);
    prev_t_ = now;
    char fpsbuf[64];
    snprintf(fpsbuf, sizeof(fpsbuf), "FPS: %.1f", fps);
    cv::putText(annotated, fpsbuf, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX,
                0.8, COL_WHITE, 2);
    cv::putText(annotated, std::string("S: ") + STATES[(int)state_],
                cv::Point(120, 25), cv::FONT_HERSHEY_SIMPLEX, 0.8, COL_WHITE,
                2);

    // Publish image as rgb8 (to mirror Python)
    auto msg_img =
        cv_bridge::CvImage(std_msgs::msg::Header(), "rgb8", annotated)
            .toImageMsg();
    image_pub_->publish(*msg_img);

    publish_detections(detections, contour_center, depth_img);
  }

  rs2::depth_frame post_process_depth_frame(const rs2::depth_frame &in) {
    rs2::frame f = in;
    f = to_disparity_.process(f);
    f = spatial_filter_.process(f);
    f = temporal_filter_.process(f);
    f = from_disparity_.process(f);
    f = hole_filling_filter_.process(f);
    return f.as<rs2::depth_frame>();
  }

  cv::Mat build_depth_mask(const cv::Mat &depth_image,
                           const cv::Rect &roi_bbox) {
    if (depth_image.empty())
      return cv::Mat();

    cv::Mat mask =
        (depth_image > 0) & (depth_image <= (int)depth_mask_threshold_mm_);
    mask.convertTo(mask, CV_8U, 255);

    std::vector<uint16_t> roi_valid;
    roi_valid.reserve(depth_image.total());
    auto collect = [&](const cv::Rect &r) {
      cv::Rect rc = r & cv::Rect(0, 0, depth_image.cols, depth_image.rows);
      if (rc.width <= 0 || rc.height <= 0)
        return;
      for (int y = rc.y; y < rc.y + rc.height; ++y) {
        const uint16_t *row = depth_image.ptr<uint16_t>(y);
        for (int x = rc.x; x < rc.x + rc.width; ++x) {
          uint16_t v = row[x];
          if (v > 0)
            roi_valid.push_back(v);
        }
      }
    };

    if (roi_bbox.area() > 0)
      collect(roi_bbox);
    if (roi_valid.empty()) {
      int h = depth_image.rows, w = depth_image.cols;
      collect(cv::Rect(w / 4, h / 4, w / 2, h / 2));
    }
    if (roi_valid.empty()) {
      collect(cv::Rect(0, 0, depth_image.cols, depth_image.rows));
    }
    if (roi_valid.empty())
      return mask;

    // percentile
    std::nth_element(
        roi_valid.begin(),
        roi_valid.begin() +
            (size_t)(roi_valid.size() * (depth_mask_percentile_ / 100.0)),
        roi_valid.end());
    uint16_t d_min = roi_valid[(size_t)(roi_valid.size() *
                                        (depth_mask_percentile_ / 100.0))];
    d_min = std::min<uint16_t>(d_min, (uint16_t)depth_mask_threshold_mm_);
    double margin = std::max(0.0, depth_mask_margin_mm_);

    cv::Mat dynamic_mask =
        (depth_image >= d_min) & (depth_image <= (uint16_t)(d_min + margin));
    dynamic_mask.convertTo(dynamic_mask, CV_8U, 255);

    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, cv::Size(depth_kernel_size_, depth_kernel_size_));
    cv::morphologyEx(dynamic_mask, dynamic_mask, cv::MORPH_CLOSE, kernel);
    cv::dilate(dynamic_mask, dynamic_mask, kernel, cv::Point(-1, -1), 1);
    cv::morphologyEx(dynamic_mask, dynamic_mask, cv::MORPH_CLOSE, kernel);

    cv::Mat gated = cv::Mat::zeros(dynamic_mask.size(), dynamic_mask.type());
    if (roi_bbox.area() > 0) {
      cv::Rect rc =
          roi_bbox & cv::Rect(0, 0, dynamic_mask.cols, dynamic_mask.rows);
      if (rc.area() > 0)
        dynamic_mask(rc).copyTo(gated(rc));
      return gated;
    }
    return dynamic_mask;
  }

  std::optional<std::pair<cv::Point, std::vector<cv::Point>>>
  find_depth_contour(const cv::Mat &depth_mask, const cv::Rect &bbox) {
    if (depth_mask.empty())
      return std::nullopt;
    cv::Mat search_mask = depth_mask.clone();
    if (bbox.area() > 0) {
      cv::Mat gated = cv::Mat::zeros(search_mask.size(), search_mask.type());
      cv::Rect rc = bbox & cv::Rect(0, 0, search_mask.cols, search_mask.rows);
      if (rc.area() <= 0)
        return std::nullopt;
      search_mask(rc).copyTo(gated(rc));
      search_mask = gated;
    }

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(search_mask, contours, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty())
      return std::nullopt;
    auto best_it = std::max_element(
        contours.begin(), contours.end(), [](auto &a, auto &b) {
          return cv::contourArea(a) < cv::contourArea(b);
        });
    double area = cv::contourArea(*best_it);
    if (area < depth_min_contour_area_)
      return std::nullopt;
    cv::Moments M = cv::moments(*best_it);
    if (M.m00 == 0.0)
      return std::nullopt;
    cv::Point c((int)(M.m10 / M.m00), (int)(M.m01 / M.m00));
    return std::make_optional(std::make_pair(c, *best_it));
  }

  cv::Rect expand_bbox(const cv::Rect2f &bbox, double margin_ratio,
                       const cv::Size &imgsz) {
    double mx = bbox.width * margin_ratio;
    double my = bbox.height * margin_ratio;
    int x1 = std::max(0, (int)std::floor(bbox.x - mx));
    int y1 = std::max(0, (int)std::floor(bbox.y - my));
    int x2 =
        std::min(imgsz.width - 1, (int)std::ceil(bbox.x + bbox.width + mx));
    int y2 =
        std::min(imgsz.height - 1, (int)std::ceil(bbox.y + bbox.height + my));
    if (x2 <= x1 || y2 <= y1)
      return cv::Rect();
    return cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
  }

  cv::Rect bbox_from_center(const cv::Point &center, int half_size,
                            const cv::Size &imgsz) {
    int x1 = std::max(0, center.x - half_size);
    int y1 = std::max(0, center.y - half_size);
    int x2 = std::min(imgsz.width - 1, center.x + half_size);
    int y2 = std::min(imgsz.height - 1, center.y + half_size);
    if (x2 <= x1 || y2 <= y1)
      return cv::Rect();
    return cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
  }

  std::optional<float> get_depth_at(const cv::Mat &depth_array, float cx,
                                    float cy) {
    if (depth_array.empty())
      return std::nullopt;
    int w = depth_array.cols, h = depth_array.rows;
    int x = std::clamp((int)std::lround(cx), 0, w - 1);
    int y = std::clamp((int)std::lround(cy), 0, h - 1);
    uint16_t v = depth_array.at<uint16_t>(y, x);
    if (v == 0)
      return std::nullopt;
    return (float)v * (float)depth_scale_ /
           1000.0f; // mm to meters if scale is mm->m factor
  }

  std::string class_name_from_id(int cid) const {
    for (const auto &kv : COCO_CLASSES)
      if (kv.first == cid)
        return kv.second;
    return std::string("unknown");
  }

  void publish_detections(const std::vector<Detection> &detections,
                          const std::optional<cv::Point> &contour_center,
                          const cv::Mat &depth_array) {
    std_msgs::msg::Float64MultiArray msg;
    msg.data.reserve(detections.size() * 4 + (contour_center ? 4 : 0));
    for (const auto &d : detections) {
      double depth_val = d.depth.has_value()
                             ? (double)d.depth.value()
                             : std::numeric_limits<double>::quiet_NaN();
      msg.data.push_back((double)d.class_id);
      msg.data.push_back((double)d.center.x);
      msg.data.push_back((double)d.center.y);
      msg.data.push_back(depth_val);
    }
    if (contour_center) {
      auto depth = get_depth_at(depth_array, (float)contour_center->x,
                                (float)contour_center->y);
      double depth_val = depth.has_value()
                             ? (double)depth.value()
                             : std::numeric_limits<double>::quiet_NaN();
      msg.data.push_back((double)target_class_id_);
      msg.data.push_back((double)contour_center->x);
      msg.data.push_back((double)contour_center->y);
      msg.data.push_back(depth_val);
    }
    target_pub_->publish(msg);
  }

  static double now_sec() {
    using clock = std::chrono::steady_clock;
    static auto t0 = clock::now();
    auto dt = std::chrono::duration<double>(clock::now() - t0).count();
    return dt;
  }

private:
  // Parameters
  std::string model_path_;
  std::string target_class_name_;
  int target_class_id_;
  double depth_scale_;

  double finish_time_;
  double depth_mask_threshold_mm_;
  double depth_mask_margin_mm_;
  double depth_mask_percentile_;
  int depth_kernel_size_;
  double depth_min_contour_area_;
  double track_window_px_;
  double search_bbox_margin_;
  double depth_threshold_mm_;
  std::string bag_path_;
  double frame_rate_;

  std::vector<int> target_labels_;

  // Model
  std::unique_ptr<RFDETR_ONNX> model_;
  bool use_cuda_param_ = true;

  // ROS
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr target_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr servo_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr yaw_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr state_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  // State
  State state_;
  State prev_state_;
  double prev_t_;
  std::optional<double> track_stable_start_;
  std::optional<cv::Point> prev_contour_center_;
  int detect_yes_ = 0, detect_no_ = 0;
  const int DETECT_YES_THRESH_ = 3;
  const int DETECT_NO_THRESH_ = 5;
  int contour_yes_ = 0, contour_no_ = 0;
  const int CONTOUR_YES_THRESH_ = 3;
  const int CONTOUR_NO_THRESH_ = 5;

  // Control
  double pitch_min_, pitch_max_;
  double pitch_angle_, pitch_angle_prev_;
  double yaw_angle_, yaw_angle_prev_;

  int image_h_, image_w_;
  double image_cx_, image_cy_;
  int x_max_, y_max_;

  // RealSense
  rs2::pipeline pipe_;
  rs2::config config_;
  std::unique_ptr<rs2::align> align_;
  rs2::spatial_filter spatial_filter_;
  rs2::temporal_filter temporal_filter_;
  rs2::hole_filling_filter hole_filling_filter_;
  rs2::disparity_transform to_disparity_{true};
  rs2::disparity_transform from_disparity_{false};
};

} // anonymous namespace

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<RFDetrNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
