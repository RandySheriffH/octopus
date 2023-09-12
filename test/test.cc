#include "octopus/threadpool.h"
#include <oneapi/tbb.h>
#include <iostream>
#include <vector>
#include <thread>
#include <bitset>
#include <cassert>
#include <cmath>
#include <map>

#ifdef _WIN32
#include <windows.h>
#undef max

std::uint64_t SubtractFILETIME(const FILETIME& ft_a, const FILETIME& ft_b) {
    LARGE_INTEGER a, b;
    a.LowPart = ft_a.dwLowDateTime;
    a.HighPart = ft_a.dwHighDateTime;
    b.LowPart = ft_b.dwLowDateTime;
    b.HighPart = ft_b.dwHighDateTime;
    return a.QuadPart - b.QuadPart;
}

class CPUUsage {
public:
    CPUUsage() {
        Reset();
    }

    short GetUsage() const {
        FILETIME sys_idle_ft, sys_kernel_ft, sys_user_ft, proc_creation_ft, proc_exit_ft, proc_kernel_ft, proc_user_ft;
        GetSystemTimes(&sys_idle_ft, &sys_kernel_ft, &sys_user_ft);
        GetProcessTimes(GetCurrentProcess(), &proc_creation_ft, &proc_exit_ft, &proc_kernel_ft, &proc_user_ft);
        std::uint64_t sys_kernel_ft_diff = SubtractFILETIME(sys_kernel_ft, sys_kernel_ft_);
        std::uint64_t sys_user_ft_diff = SubtractFILETIME(sys_user_ft, sys_user_ft_);
        std::uint64_t proc_kernel_diff = SubtractFILETIME(proc_kernel_ft, proc_kernel_ft_);
        std::uint64_t proc_user_diff = SubtractFILETIME(proc_user_ft, proc_user_ft_);
        std::uint64_t total_sys = sys_kernel_ft_diff + sys_user_ft_diff;
        std::uint64_t total_proc = proc_kernel_diff + proc_user_diff;
        return total_sys > 0 ? static_cast<double>(total_proc) / total_sys * 100.0 : 0;
    }

    void Reset() {
        FILETIME sys_idle_ft, proc_creation_ft, proc_exit_ft;
        GetSystemTimes(&sys_idle_ft, &sys_kernel_ft_, &sys_user_ft_);
        GetProcessTimes(GetCurrentProcess(), &proc_creation_ft, &proc_exit_ft, &proc_kernel_ft_, &proc_user_ft_);
    }

private:
    //system total times
    FILETIME sys_kernel_ft_;
    FILETIME sys_user_ft_;

    //process times
    FILETIME proc_kernel_ft_;
    FILETIME proc_user_ft_;
};

#else

#include <fstream>
#include <numeric>
#include <unistd.h>

class CPUUsage {
public:
    CPUUsage() {
        Reset();
    }

    short GetUsage() {
        size_t idle_time{}, total_time{};
        get_cpu_times(idle_time, total_time);
        const float idle_time_delta = idle_time - previous_idle_time_;
        const float total_time_delta = total_time - previous_total_time_;
        const float utilization = 100.0 * (1.0 - idle_time_delta / total_time_delta);
        previous_idle_time_ = idle_time;
        previous_total_time_ = total_time;
        return (short)utilization;
    }

    void Reset() {
        GetUsage();
    }

    static std::vector<size_t> get_cpu_times() {
        std::ifstream proc_stat("/proc/stat");
        proc_stat.ignore(5, ' '); // Skip the 'cpu' prefix.
        std::vector<size_t> times;
        for (size_t time; proc_stat >> time; times.push_back(time));
        return times;
    }

    bool get_cpu_times(size_t& idle_time, size_t& total_time) {
        const std::vector<size_t> cpu_times = get_cpu_times();
        if (cpu_times.size() < 4)
            return false;
        idle_time = cpu_times[3];
        total_time = std::accumulate(cpu_times.begin(), cpu_times.end(), 0);
        return true;
    }

private:

    size_t previous_idle_time_{};
    size_t previous_total_time_{};
};

#endif

////////////////////////////////////////////////////////////////////////////////////////

template<size_t CAPACITY, size_t SCALE>
void TestQueue() {
    assert(SCALE > 0);

    struct Tick {
        int64_t tick = std::numeric_limits<int64_t>::max();
    };

    auto main_thread_pid = std::this_thread::get_id();
    std::map<size_t, std::thread::id> bit_thread_map;
    for (size_t i = 0; i < SCALE; ++i) {
        bit_thread_map[i] = main_thread_pid;
    }

    octopus::Iter topdown_iter = SCALE - 1;
    octopus::Iter bottomup_iter = 0;
    octopus::Queue<Tick, CAPACITY> tick_queue;

    std::vector<std::thread> threads;
    auto concurrency = std::thread::hardware_concurrency();
    threads.reserve(concurrency);

    for (decltype(concurrency) i = 0; i < concurrency - 1; ++i) {
        threads.emplace_back(std::thread([&]() {
            while (true) {
                auto j = topdown_iter.load();
                if (j >= 0) {
                    auto jj = j - 1;
                    if (topdown_iter.compare_exchange_weak(j, jj)) {
                        Tick tick = { j };
                        if (!tick_queue.PushTail(std::move(tick))) {
                            assert(bit_thread_map[tick.tick] == main_thread_pid);
                            bit_thread_map[tick.tick] = std::this_thread::get_id();
                        }
                    }
                }
                else {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            while (true) {
                auto j = bottomup_iter.load();
                if (j < SCALE) {
                    auto jj = j + 1;
                    if (bottomup_iter.compare_exchange_weak(j, jj)) {
                        Tick tick = tick_queue.PopHead();
                        if (tick.tick < SCALE) {
                            assert(bit_thread_map[tick.tick] == main_thread_pid);
                            bit_thread_map[tick.tick] = std::this_thread::get_id();
                        }
                    }
                }
                else {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            }));
    }
    for (auto& thread : threads) {
        thread.join();
    }

    assert(topdown_iter.load() == -1);
    assert(bottomup_iter.load() == SCALE);

    std::map<std::thread::id, size_t> thread_bit_map;
    for (const auto& pair : bit_thread_map) {
        thread_bit_map[pair.second]++;
    }

    size_t total = 0;
    for (const auto& pair : thread_bit_map) {
        assert(pair.first != main_thread_pid);
        std::cout << "thread " << pair.first << ": " << pair.second << std::endl;
        total += pair.second;
    }

    assert(total == SCALE);
    std::cout << "total: " << total << std::endl;
    std::cout << "TestQueue done." << std::endl;
}

template<size_t CAPACITY, size_t SCALE>
void TestTpIntegrity() {
    {
        octopus::ThreadPool tp(std::thread::hardware_concurrency());

        std::atomic<size_t> total = 0;
        std::thread thread = std::thread([&]() {
            octopus::Fn fn = [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
                total.fetch_add(end - begin);
                };
            tp.ParallFor(fn, SCALE);
            });
        thread.join();
        assert(total == SCALE);

        std::vector<std::atomic<size_t>> totals(CAPACITY);
        std::vector<std::thread> threads;
        threads.reserve(CAPACITY);
        for (size_t i = 0; i < CAPACITY; ++i) {
            totals[i] = 0;
            threads.emplace_back(std::thread([&, i]() {
                octopus::Fn fn = [&, i](std::ptrdiff_t begin, std::ptrdiff_t end) {
                    totals[i].fetch_add(end - begin);
                    };
                tp.ParallFor(&fn, SCALE);
                }));
        }
        for (auto& thread : threads) {
            thread.join();
        }
        for (size_t i = 0; i < CAPACITY; ++i) {
            OCT_ENFORCE(totals[i].load() == SCALE, "");
        }
    }

    std::cout << "TestTpIntegrity done." << std::endl;
}

template<size_t SCALE, size_t SCALE2>
void TestSubThreadEmdded() {

    octopus::ThreadPool tp(std::thread::hardware_concurrency());

    std::unique_ptr<float[]> A{ new float[SCALE * SCALE2] };
    std::unique_ptr<float[]> B{ new float[SCALE * SCALE2] };
    std::unique_ptr<float[]> C{ new float[SCALE * SCALE2] };

    std::unique_ptr<std::thread::id[]> thread_ids{ new std::thread::id[SCALE] };

    constexpr float a = 12345;
    constexpr float b = 54321;
    constexpr float c = a + b;

    for (size_t i = 0; i < SCALE * SCALE2; ++i) {
        A[i] = a;
        B[i] = b;
    }

    std::cout << "Main tid: " << std::this_thread::get_id() << std::endl;

    octopus::Fn fn2 = [&](std::ptrdiff_t begin2, std::ptrdiff_t end2) {
        for (auto i = begin2; i < end2; ++i) {
            C[i] = A[i] + B[i];
        }
        };

    octopus::Fn fn = [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
        auto tid = std::this_thread::get_id();
        for (auto i = begin; i < end; ++i) {
            thread_ids[i] = tid;
        }
        tp.ParallFor(&fn2, begin * SCALE2, end * SCALE2);
        };

    auto tm_start = std::chrono::steady_clock::now();
    tp.ParallFor(fn, SCALE);
    auto tm_stop = std::chrono::steady_clock::now();

    std::unordered_map<std::thread::id, size_t> counter;
    for (size_t i = 0; i < SCALE; ++i) {
        counter[thread_ids[i]]++;
    }
    for (const auto& pair : counter) {
        std::cout << "thread " << pair.first << ": " << pair.second << std::endl;
    }
    for (size_t i = 0; i < SCALE * SCALE2; ++i) {
        OCT_ENFORCE(std::abs(C[i] - c) < 1e-5, "");
    }
    std::cout << "In " << std::chrono::duration_cast<std::chrono::milliseconds>(tm_stop - tm_start).count() << " ms" << std::endl;
    std::cout << "TestSubThreadEmdded done." << std::endl;
}

enum TpType {
    oct_t = 0,
    tbb_t,
    omp_t,
};

using SimpleFn = std::function<void(std::ptrdiff_t)>;

struct Tp {
    virtual void ParallelFor(std::ptrdiff_t total, const SimpleFn& simple_fn) = 0;
    static std::unique_ptr<Tp> GetInstance(TpType tp_type);
};

struct TpOct : public Tp {
    TpOct() : hw_(std::thread::hardware_concurrency()), tp_(hw_) {}
    void ParallelFor(std::ptrdiff_t total, const SimpleFn& simple_fn) override {
        octopus::Fn fn = [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
            for (auto ith = begin; ith < end; ++ith) {
                simple_fn(ith);
            }
            };
        //octopus::BinaryPartitioner partitioner(20000);
        octopus::BinaryPartitioner partitioner(1);
        //octopus::BinaryPartitioner partitioner(10);
        //auto granularity = std::floor(std::pow(total, 1.0 / hw_));
        //octopus::BinaryPartitioner partitioner(static_cast<std::ptrdiff_t>(granularity));
        tp_.ParallFor(fn, total, &partitioner);
    }
    unsigned int hw_;
    octopus::ThreadPool tp_;
};

struct TbbTask {
    TbbTask(const octopus::Fn& fn) : fn_(fn) {}
    void operator()(const oneapi::tbb::blocked_range<std::ptrdiff_t>& r) const {
        auto begin = r.begin();
        auto end = r.end();
        fn_(begin, end);
    }
    const octopus::Fn& fn_;
};

struct TpTbb : public Tp {
    TpTbb() {
        tbb_gobal_ =
            std::make_unique<oneapi::tbb::global_control>(
                oneapi::tbb::global_control::max_allowed_parallelism,
                std::thread::hardware_concurrency());
    }
    void ParallelFor(std::ptrdiff_t total, const SimpleFn& simple_fn) override {
        octopus::Fn fn = [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
            for (auto ith = begin; ith < end; ++ith) {
                simple_fn(ith);
            }
            };

        TbbTask tbb_task(fn);
        oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<std::ptrdiff_t>(0, total, 1), tbb_task);
    }

    std::unique_ptr<oneapi::tbb::global_control> tbb_gobal_;
};

struct TpOmp : public Tp {
    void ParallelFor(std::ptrdiff_t total, const SimpleFn& simple_fn) override {
        long long total_size = static_cast<long long>(total);
#pragma omp parallel for
        for (long long ith = 0; ith < total_size; ++ith) {
            simple_fn(static_cast<std::ptrdiff_t>(ith));
        }
    }
};

std::unique_ptr<Tp> Tp::GetInstance(TpType tp_type) {
    if (tp_type == oct_t) {
        return std::make_unique<TpOct>();
    }
    else if (tp_type == tbb_t) {
        return std::make_unique<TpTbb>();
    }
    else if (tp_type == omp_t) {
        return std::make_unique<TpOmp>();
    }
    else {
        throw std::invalid_argument("unknown tp type!");
    }
}


template<size_t SCALE, std::ptrdiff_t REPEAT> // so total computation would be SCALE * REPEAT
void TestTpPerf(TpType tp_type, const std::string& tp_name) {

    std::unique_ptr<float[]> A_ptr = std::make_unique<float[]>(SCALE);
    std::unique_ptr<float[]> B_ptr = std::make_unique<float[]>(SCALE);
    std::unique_ptr<float[]> C_ptr = std::make_unique<float[]>(SCALE);
    float* A = A_ptr.get();
    float* B = B_ptr.get();
    float* C = C_ptr.get();

    std::unique_ptr <std::thread::id[]> thread_ids_ptr = std::make_unique<std::thread::id[]>(SCALE);
    std::thread::id* thread_ids = thread_ids_ptr.get();

    constexpr float a = 12;
    constexpr float b = 53;
    constexpr float c = a * b * REPEAT;

    for (size_t i = 0; i < SCALE; ++i) {
        A[i] = a;
        B[i] = b;
        C[i] = 0;
    }

    std::cout << "Main tid: " << std::this_thread::get_id() << std::endl;

    auto simple_fn = [thread_ids, A, B, C](std::ptrdiff_t ith) {
        thread_ids[ith] = std::this_thread::get_id();
        for (auto repeat = 0; repeat < REPEAT; ++repeat) {
            C[ith] += A[ith] * B[ith];
        }
        };

    auto tp = Tp::GetInstance(tp_type);
    assert(tp.get());

    CPUUsage cpu_usage;
    auto tm_start = std::chrono::steady_clock::now();
    tp->ParallelFor(SCALE, simple_fn);
    auto tm_stop = std::chrono::steady_clock::now();
    auto cpu_usage_percentage = cpu_usage.GetUsage();

    size_t breaks = 0;
    std::unordered_map<std::thread::id, size_t> counter;
    for (size_t i = 0; i < SCALE; ++i) {
        counter[thread_ids[i]]++;
        if (i + 1 < SCALE && thread_ids[i] != thread_ids[i + 1]) {
            ++breaks;
        }
    }
    for (const auto& pair : counter) {
        std::cout << "thread " << pair.first << ": " << pair.second << std::endl;
    }
    std::cout << "breaks: " << breaks << std::endl;
    for (size_t i = 0; i < SCALE; ++i) {
        OCT_ENFORCE(std::abs(C[i] - c) < 1e-5, "result mismatch at " + std::to_string(i) + ", " + std::to_string(C[i]));
    }
    std::cout << "In " << std::chrono::duration_cast<std::chrono::milliseconds>(tm_stop - tm_start).count() << " ms" << std::endl;
    std::cout << "Cpu usage: " << cpu_usage_percentage << "%" << std::endl;
    std::cout << "Test " << tp_name << " done." << std::endl;
}

#define BREAK std::cout << "----------------------" << std::endl

int main() {
    std::cout << "hi, Mr Octopus!" << std::endl;
    try {
        TestQueue<64, 1000>();
        BREAK;
        TestTpIntegrity<10, 2000>();
        BREAK;
        TestSubThreadEmdded<10000, 100>();
        BREAK;
        TestTpPerf<10000000, 100>(omp_t, "OMP");
        BREAK;
        TestTpPerf<10000000, 100>(tbb_t, "TBB");
        BREAK;
        TestTpPerf<10000000, 100>(oct_t, "OCT");
        BREAK;
    }
    catch (std::exception& ex) {
        std::cout << "caught exception: " << ex.what() << std::endl;
        BREAK;
    }
    std::cout << "see ya, Mr Octopus!" << std::endl;
}

