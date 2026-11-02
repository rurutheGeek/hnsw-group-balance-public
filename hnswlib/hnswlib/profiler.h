#pragma once

#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <atomic>
#include <thread>
#include <mutex>
#include <iostream>
#include <sstream>

namespace hnswlib {

enum EventType : uint8_t {
    EVENT_NONE = 0,
    // Search Events
    SEARCH_START = 1,
    SEARCH_END = 2,
    LAYER_SEARCH_START = 3,
    LAYER_SEARCH_END = 4,
    BASE_LAYER_SEARCH_START = 5,
    BASE_LAYER_SEARCH_END = 6,
    HOP_ITERATION = 7,
    DIST_CALC_BATCH = 8,
    
    // Build (Insert) Events
    INSERT_START = 10,
    INSERT_END = 11,
    LEVEL_DECIDED = 12,
    INSERT_ZOOM_IN_START = 13,
    INSERT_ZOOM_IN_END = 14,
    INSERT_LAYER_PROCESS_START = 15,
    INSERT_LAYER_PROCESS_END = 16,
    SELECT_NEIGHBORS_START = 17,
    SELECT_NEIGHBORS_END = 18,
    CONNECT_NEIGHBORS_START = 19,
    CONNECT_NEIGHBORS_END = 20,
    
    // New Insert Details
    UPDATE_ENTRY_POINT_START = 21,
    UPDATE_ENTRY_POINT_END = 22,
    INSERT_SEARCH_CANDIDATES_START = 23,
    INSERT_SEARCH_CANDIDATES_END = 24
};

struct LogEntry {
    uint64_t timestamp;
    uint64_t payload;
    uint64_t thread_id;
    uint8_t event_type;
};

const size_t MAX_BUFFER_SIZE = 10000000; 

class GlobalProfiler {
public:
    std::vector<LogEntry> buffer_;
    std::mutex mutex_;
    bool enabled_ = false;

    static GlobalProfiler& instance() {
        static GlobalProfiler instance;
        return instance;
    }

    GlobalProfiler() {
        buffer_.reserve(MAX_BUFFER_SIZE);
    }

    void setEnabled(bool enable) {
        std::lock_guard<std::mutex> lock(mutex_);
        enabled_ = enable;
        if (enable) {
            buffer_.clear();
        }
    }

    bool isEnabled() const {
        return enabled_;
    }

    void log(EventType type, uint64_t payload) {
        if (!enabled_) return;

        std::lock_guard<std::mutex> lock(mutex_);
        if (enabled_ && buffer_.size() < MAX_BUFFER_SIZE) {
            uint64_t ts = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            uint64_t tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
            buffer_.push_back({ts, payload, tid, type});
        }
    }

    std::string dump() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::stringstream ss;
        ss << "thread_id,timestamp,event_type,payload\n";
        
        for (const auto& e : buffer_) {
            ss << e.thread_id << "," << e.timestamp << "," << (int)e.event_type << "," << e.payload << "\n";
        }
        return ss.str();
    }
};

#define PROFILER_LOG(type, payload) \
    hnswlib::GlobalProfiler::instance().log(type, payload);

} // namespace hnswlib
