#include "util/event.hpp"

#include <chrono>

namespace sstvae::util {

void Event::set() {
    {
        std::lock_guard<std::mutex> lock(m_);
        set_ = true;
    }
    cv_.notify_all();
}

void Event::clear() {
    std::lock_guard<std::mutex> lock(m_);
    set_ = false;
}

bool Event::is_set() const {
    std::lock_guard<std::mutex> lock(m_);
    return set_;
}

bool Event::wait(double seconds) {
    std::unique_lock<std::mutex> lock(m_);
    if (set_) return true;
    cv_.wait_for(lock, std::chrono::duration<double>(seconds), [&] { return set_; });
    return set_;
}

}  // namespace sstvae::util
