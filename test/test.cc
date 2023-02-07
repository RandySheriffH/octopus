#include "octopus/threadpool.h"
#include <oneapi/tbb.h>
#include <iostream>
#include <vector>
#include <thread>
#include <bitset>
#include <cassert>
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

	double GetUsage() const {
		FILETIME sys_idle_ft, sys_kernel_ft, sys_user_ft, proc_creation_ft, proc_exit_ft, proc_kernel_ft, proc_user_ft;
		GetSystemTimes(&sys_idle_ft, &sys_kernel_ft, &sys_user_ft);
		GetProcessTimes(GetCurrentProcess(), &proc_creation_ft, &proc_exit_ft, &proc_kernel_ft, &proc_user_ft);
		std::uint64_t sys_kernel_ft_diff = SubtractFILETIME(sys_kernel_ft, sys_kernel_ft_);
		std::uint64_t sys_user_ft_diff = SubtractFILETIME(sys_user_ft, sys_user_ft_);
		std::uint64_t proc_kernel_diff = SubtractFILETIME(proc_kernel_ft, proc_kernel_ft_);
		std::uint64_t proc_user_diff = SubtractFILETIME(proc_user_ft, proc_user_ft_);
		std::uint64_t total_sys = sys_kernel_ft_diff + sys_user_ft_diff;
		std::uint64_t total_proc = proc_kernel_diff + proc_user_diff;
		return total_sys > 0 ? static_cast<double>(total_proc) / total_sys : 0;
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

#include <cstddef>
#include <sys/times.h>
#include <sys/resource.h>

class CPUUsage : public ICPUUsage {
 public:
  CPUUsage() {
    Reset();
  }

  short GetUsage() const override {
    struct tms time_sample;
    clock_t total_clock_now = times(&time_sample);
    if (total_clock_now <= total_clock_start_ ||
        time_sample.tms_stime < proc_sys_clock_start_ ||
        time_sample.tms_utime < proc_user_clock_start_) {
      // overflow detection
      return -1;
    } else {
      clock_t proc_total_clock_diff = (time_sample.tms_stime - proc_sys_clock_start_) + (time_sample.tms_utime - proc_user_clock_start_);
      clock_t total_clock_diff = total_clock_now - total_clock_start_;
      return static_cast<short>(100.0 * proc_total_clock_diff / total_clock_diff / onnxruntime::Env::Default().GetNumPhysicalCpuCores());
    }
  }

  void Reset() override {
    struct tms time_sample;
    total_clock_start_ = times(&time_sample);
    proc_sys_clock_start_ = time_sample.tms_stime;
    proc_user_clock_start_ = time_sample.tms_utime;
  }

 private:
  clock_t total_clock_start_;
  clock_t proc_sys_clock_start_;
  clock_t proc_user_clock_start_;
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

template<size_t THREAD, size_t CAPACITY, size_t SCALE>
void TestMainThread() {

	octopus::ThreadPool tp(THREAD);

	std::atomic<size_t> total = 0;
	std::thread thread = std::thread([&]() {
		octopus::Fn fn = [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
			total.fetch_add(end - begin);
		};
		tp.ParallFor(&fn, SCALE);
		});
	thread.join();
	assert(total == SCALE);

	std::vector<std::atomic<size_t>> totals(CAPACITY);
	std::vector<std::thread> threads;
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

	std::cout << "TestMainThread done." << std::endl;
}

template<size_t THREAD, size_t SCALE, std::ptrdiff_t REPEAT>
void TestSubThread() {

	octopus::ThreadPool tp(THREAD);

	std::unique_ptr<float[]> A = std::make_unique<float[]>(SCALE);
	std::unique_ptr<float[]> B = std::make_unique<float[]>(SCALE);
	std::unique_ptr<float[]> C = std::make_unique<float[]>(SCALE);

	std::unique_ptr <std::thread::id[]> thread_ids = std::make_unique<std::thread::id[]>(SCALE);

	constexpr float a = 12;
	constexpr float b = 53;
	constexpr float c = a * b * REPEAT;

	for (size_t i = 0; i < SCALE; ++i) {
		A[i] = a;
		B[i] = b;
		C[i] = 0;
	}

	std::cout << "Main tid: " << std::this_thread::get_id() << std::endl;

	octopus::Fn fn = [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
		auto tid = std::this_thread::get_id();
		for (auto i = begin; i < end; ++i) {
			for (auto j = 0; j < REPEAT; ++j) {
				C[i] += A[i] * B[i];
			}
			thread_ids[i] = tid;
		}
	};

	octopus::AffinityPartitioner partitioner(THREAD, std::max(SCALE / (10 * THREAD), 1ULL));
	//octopus::StaticPartitioner partitioner(std::max(static_cast<std::ptrdiff_t>(SCALE)/(10*THREAD), 1ULL));
	//octopus::AffinityPartitioner partitioner(2, SCALE/(THREAD*10));

	CPUUsage cpu_usage;
	auto tm_start = std::chrono::steady_clock::now();
	tp.ParallFor(&fn, SCALE, &partitioner);
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
		OCT_ENFORCE(std::abs(C[i] - c) < 1e-5, "");
	}
	std::cout << "In " << std::chrono::duration_cast<std::chrono::milliseconds>(tm_stop - tm_start).count() << " ms" << std::endl;
	std::cout << "Cpu usage: " << cpu_usage_percentage * 100 << "%" << std::endl;
	std::cout << "TestSubThread done." << std::endl;
}

template<size_t THREAD, size_t SCALE, size_t SCALE2>
void TestSubThreadEmdded() {

	octopus::ThreadPool tp(THREAD);

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
	tp.ParallFor(&fn, SCALE);
	auto tm_stop = std::chrono::steady_clock::now();

	std::unordered_map<std::thread::id, size_t> counter;
	for (size_t i = 0; i < SCALE; ++i) {
		counter[thread_ids[i]]++;
	}
	for (const auto& pair : counter) {
		std::cout << "thread " << pair.first << ": " << pair.second << std::endl;
	}
	for (size_t i = 0; i < SCALE * SCALE2; ++i) {
		OCT_ENFORCE(std::abs(C[i]-c) < 1e-5, "");
	}
	std::cout << "In " << std::chrono::duration_cast<std::chrono::milliseconds>(tm_stop - tm_start).count() << " ms" << std::endl;
	std::cout << "TestSubThreadEmdded done." << std::endl;
}

struct TbbTask {
	TbbTask(const octopus::Fn& fn) : fn_(fn) {}
	void operator()(const oneapi::tbb::blocked_range<std::ptrdiff_t>& r) const {
		auto begin = r.begin();
		auto end = r.end();
		fn_(begin, end);
	}
	const octopus::Fn& fn_;
};

template<size_t CAPATICY, size_t SCALE, std::ptrdiff_t REPEAT>
void TestTBB() {

	std::unique_ptr<float[]> A = std::make_unique<float[]>(SCALE);
	std::unique_ptr<float[]> B = std::make_unique<float[]>(SCALE);
	std::unique_ptr<float[]> C = std::make_unique<float[]>(SCALE);

	std::unique_ptr <std::thread::id[]> thread_ids = std::make_unique<std::thread::id[]>(SCALE);

	constexpr float a = 12;
	constexpr float b = 53;
	constexpr float c = a * b * REPEAT;

	for (size_t i = 0; i < SCALE; ++i) {
		A[i] = a;
		B[i] = b;
		C[i] = 0;
	}

	std::cout << "Main tid: " << std::this_thread::get_id() << std::endl;

	octopus::Fn fn = [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
		auto tid = std::this_thread::get_id();
		for (auto i = begin; i < end; ++i) {
			for (auto j = 0; j < REPEAT; ++j) {
				C[i] += A[i] * B[i];
			}
			thread_ids[i] = tid;
		}
	};

	auto tbb_global =
		std::make_unique<oneapi::tbb::global_control>(
			oneapi::tbb::global_control::max_allowed_parallelism,
			CAPATICY);

	TbbTask tbb_task(fn);
	CPUUsage cpu_usage;
	auto tm_start = std::chrono::steady_clock::now();
	oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<std::ptrdiff_t>(0, SCALE, 1), tbb_task);
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
		OCT_ENFORCE(std::abs(C[i] - c) < 1e-5, "");
	}
	std::cout << "In " << std::chrono::duration_cast<std::chrono::milliseconds>(tm_stop - tm_start).count() << " ms" << std::endl;
	std::cout << "Cpu usage: " << cpu_usage_percentage * 100 << "%" << std::endl;
	std::cout << "TestTBB done." << std::endl;
}

#define BREAK std::cout << "----------------------" << std::endl

int main() {
	std::cout << "hi, Mr Octopus!" << std::endl;
	//TestQueue<64, 1000>();
	//BREAK;
	//TestMainThread<4, 10, 2000>();
	//BREAK;
	TestSubThread<8, 10000000, 100>();
	BREAK;
	//TestSubThreadEmdded<4, 10000, 100>();
	//BREAK;
	TestTBB<8, 10000000, 100>();
}