#include "booster_vision/base/data_syncer.hpp"

#include <filesystem>
#include <stdexcept>
#include <regex>
#include <algorithm>
#include <iostream>
#include <cfloat>

namespace booster_vision {

void DataSyncer::LoadData(const std::string &data_dir) {
    if (!std::filesystem::is_directory(data_dir)) {
        throw std::runtime_error("data directory does not exist: " + data_dir);
    }
    data_dir_ = data_dir;
    
    std::vector<std::string> files;
    for (const auto &entry : std::filesystem::directory_iterator(data_dir)) {
        files.push_back(entry.path().filename().string());
    }

    std::regex color_file_regex("color_([0-9]+\\.[0-9]+)\\.jpg");
    std::smatch m;
    time_stamp_list_.clear();

    for (const std::string &file : files) {
        if (std::regex_match(file, m, color_file_regex)) {
            double timestamp = std::stod(m[1].str());
            time_stamp_list_.push_back(timestamp);
        }
    }
    
    std::sort(time_stamp_list_.begin(), time_stamp_list_.end());
    data_index_ = 0;
    std::cout << "[DataSyncer] Loaded " << time_stamp_list_.size() << " data frames." << std::endl;
}

void DataSyncer::AddDepth(const DepthDataBlock &depth_data) {
    if (!enable_depth_) return;
    std::lock_guard<std::mutex> lock(depth_buffer_mutex_);
    depth_buffer_.push_back(depth_data);
    
    // Prevent Memory Leak: Keep buffer size manageable
    if (depth_buffer_.size() > kDepthBufferLength * 2) {
        depth_buffer_.erase(depth_buffer_.begin());
    }
}

void DataSyncer::AddPose(const PoseDataBlock &pose_data) {
    std::lock_guard<std::mutex> lock(pose_buffer_mutex_);
    pose_buffer_.push_back(pose_data);
    
    // Prevent Memory Leak: Keep buffer size manageable
    if (pose_buffer_.size() > kPoseBufferLength * 2) {
        pose_buffer_.erase(pose_buffer_.begin());
    }
}

SyncedDataBlock DataSyncer::getSyncedDataBlock() {
    if (time_stamp_list_.empty()) return SyncedDataBlock();

    double timestamp = time_stamp_list_[data_index_];
    std::string ts_str = std::to_string(timestamp);
    
    std::string color_file_path = data_dir_ + "/color_" + ts_str + ".jpg";
    std::string depth_file_path = data_dir_ + "/depth_" + ts_str + ".png";
    std::string pose_file_path  = data_dir_ + "/pose_"  + ts_str + ".yaml";
    
    SyncedDataBlock synced_data;

    if (std::filesystem::exists(color_file_path)) {
        synced_data.color_data.data = cv::imread(color_file_path);
        synced_data.color_data.timestamp = timestamp;
    }

    if (enable_depth_ && std::filesystem::exists(depth_file_path)) {
        synced_data.depth_data.data = cv::imread(depth_file_path, cv::IMREAD_ANYDEPTH);
        synced_data.depth_data.timestamp = timestamp;
    }

    if (std::filesystem::exists(pose_file_path)) {
        YAML::Node pose_node = YAML::LoadFile(pose_file_path);
        if (pose_node["pose"]) {
            synced_data.pose_data.data = pose_node["pose"].as<Pose>();
        } else {
            synced_data.pose_data.data = pose_node.as<Pose>();
        }
        synced_data.pose_data.timestamp = timestamp;
    }

    data_index_ = (data_index_ + 1) % time_stamp_list_.size();
    return synced_data;
}

SyncedDataBlock DataSyncer::getSyncedDataBlock(const ColorDataBlock &color_data) {
    SyncedDataBlock synced_data;
    synced_data.color_data = color_data;

    // 1. FAST SHALLOW COPY UNDER LOCK (Restored from original to prevent blocking camera)
    DepthBuffer depth_buffer_cp;
    {
        std::lock_guard<std::mutex> lock(depth_buffer_mutex_);
        depth_buffer_cp = depth_buffer_;
    }
    PoseBuffer pose_buffer_cp;
    {
        std::lock_guard<std::mutex> lock(pose_buffer_mutex_);
        pose_buffer_cp = pose_buffer_;
    }

    double color_timestamp = color_data.timestamp;

    // 2. PROCESS DEPTH (Outside the lock to save time)
    if (enable_depth_ && !depth_buffer_cp.empty()) {
        double smallest_depth_timestamp_diff = DBL_MAX;
        
        // Fix the Segfault: Only search up to the actual size of the buffer
        size_t search_len = std::min((size_t)depth_buffer_cp.size(), (size_t)kDepthBufferLength);
        
        for (auto it = depth_buffer_cp.rbegin(); it != depth_buffer_cp.rbegin() + search_len; ++it) {
            double diff = std::abs(it->timestamp - color_timestamp);
            if (diff < smallest_depth_timestamp_diff) {
                smallest_depth_timestamp_diff = diff;
                synced_data.depth_data = *it;
                synced_data.depth_data.timestamp = it->timestamp;
                
                // RESTORED DEEP COPY: Crucial for Thread Safety!
                it->data.copyTo(synced_data.depth_data.data);
            } else {
                break;
            }
        }
    }

    // 3. PROCESS POSE (Outside the lock)
    if (!pose_buffer_cp.empty()) {
        double smallest_pose_timestamp_diff = DBL_MAX;
        
        // Fix the Segfault: Only search up to the actual size of the buffer
        size_t search_len = std::min((size_t)pose_buffer_cp.size(), (size_t)kPoseBufferLength);
        
        for (auto it = pose_buffer_cp.rbegin(); it != pose_buffer_cp.rbegin() + search_len; ++it) {
            double diff = std::abs(it->timestamp - color_timestamp);
            if (diff < smallest_pose_timestamp_diff) {
                smallest_pose_timestamp_diff = diff;
                synced_data.pose_data = *it;
            } else {
                break;
            }
        }
    }

    return synced_data;
}

} // namespace booster_vision