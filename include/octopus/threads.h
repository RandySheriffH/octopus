#pragma once

#include <unordered_map>
#include <shared_mutex>
#include <functional>
#include <algorithm>
#include <cassert>
#include <vector>
#include <atomic>
#include <mutex>

#define OCT_CACHE_LINE_SIZE 64
#define OCT_PADNN(n1,n2) const int64_t n1##n2
#define OCT_PADN(n1,n2) OCT_PADNN(n1,n2)
#define OCT_PAD OCT_PADN(__pad,__COUNTER__)
#define OCT_ATOM_ACQ std::memory_order_acquire
#define OCT_ATOM_RLX std::memory_order_relaxed
#define OCT_ATOM_REL std::memory_order_release
#define OCT_ENFORCE(cond,msg) \
	if (!cond) { \
		throw std::runtime_error(msg); \
	}

namespace octopus {

	using Iter = std::atomic_int64_t;

	template<typename T, size_t CAPACITY>
	class alignas(OCT_CACHE_LINE_SIZE) Queue {
	public:
		Queue() = default;
		Queue(const Queue& q) {
			__head.store(q.__head.load());
			__tail.store(q.__tail.load());
			memcpy(__slots, q.__slots, sizeof(T*) * CAPACITY);
		};

		T Pop() {
			while (true) {
				auto head = __head.load(OCT_ATOM_RLX);
				auto tail = __tail.load(OCT_ATOM_RLX);
				if (head >= 0 && tail >= 0) {
					if (head == tail) {
						return {}; // empty
					}
					if (__head.compare_exchange_weak(head, -head - 1, OCT_ATOM_ACQ, OCT_ATOM_RLX)) {
						T t = std::move(__slots[head]);
						auto next_head = (head + 1) % CAPACITY;
						__head.store(next_head, OCT_ATOM_RLX);
						return std::move(t);
					}
				}
			}
			return {};
		}
		bool Push(T&& t) {
			while (true) {
				auto head = __head.load(OCT_ATOM_RLX);
				auto tail = __tail.load(OCT_ATOM_RLX);
				if (head >= 0 && tail >= 0) {
					auto next_tail = (tail + 1) % CAPACITY;
					if (next_tail == head) {
						return false; // full
					}
					if (__tail.compare_exchange_weak(tail, -tail - 1, OCT_ATOM_RLX, OCT_ATOM_RLX)) {
						__slots[tail] = std::forward<T>(t);
						__tail.store(next_tail, OCT_ATOM_REL);
						return true;
					}
				}
			}
			return false;
		}
	private:
		Iter __head = {};
		Iter __tail = {};
		T __slots[CAPACITY] = {};
	};

	struct Partitioner;

	using Fn = std::function<void(std::ptrdiff_t, std::ptrdiff_t)>;

	class alignas(OCT_CACHE_LINE_SIZE) Task {
	public:
		Task() = default;
		Task(Fn* fn,
			Partitioner* partitioner,
			std::ptrdiff_t begin,
			std::ptrdiff_t end) :
			__fn(fn), __partitioner(partitioner), __begin(begin), __end(end) {}

		Task(const Task& task) = default;
		Task(Task&& task) = default;

		operator bool() const { return __fn; }
		Task& operator = (const Task& task) = default;
		Task& operator = (Task&& task) = default;

		void Run();

	private:
		Fn* __fn = {};
		Partitioner* __partitioner = {};
		std::ptrdiff_t __begin = {};
		std::ptrdiff_t __end = {};
	};

	struct Partitioner {
		virtual std::ptrdiff_t Partition(std::ptrdiff_t, std::ptrdiff_t) const = 0;
	};

	class StaticPartitioner : public Partitioner {
	public:
		StaticPartitioner(std::ptrdiff_t min_chuck_size) :
			__min_chuck_size(min_chuck_size) {

			assert(min_chuck_size);
		}

		std::ptrdiff_t Partition(std::ptrdiff_t begin, std::ptrdiff_t end) const override {
			if (end - begin > __min_chuck_size) {
				return begin + __min_chuck_size;
			}
			else {
				return end;
			}
		}

	private:
		std::ptrdiff_t __min_chuck_size;
	};

	StaticPartitioner static_partitioner(1000);
	//StaticPartitioner static_partitioner(100);
	//StaticPartitioner static_partitioner(1);

	using TaskQueue = Queue<Task, 64>;
	size_t num_thread = 0;

	struct alignas(OCT_CACHE_LINE_SIZE) ThreadData {
		bool exit = false;
		std::thread::id tid = {};
	};

	std::vector<ThreadData> thread_data_vec;
	std::vector<std::thread> threads;

	// each main thread owns a task pool
	class alignas(OCT_CACHE_LINE_SIZE) TaskPool {
	public:
		friend void InsertPool(TaskPool*);
		friend void RemovePool(TaskPool*);
		friend void ThreadEntry(size_t index);

		explicit TaskPool();
		~TaskPool();

		Task PopAt(size_t at) {
			assert(at < __task_queues.size());
			return __task_queues[at].Pop();
		}
		bool PushAt(Task&& t, size_t at) {
			assert(at < __task_queues.size());
			return __task_queues[at].Push(std::forward<Task>(t));
		}
		size_t Size() const {
			return __task_queues.size();
		}
		void IncRef(size_t at) {
			assert(at < __task_queues.size());
			__refs[at]++;
		}
		void DecRef(size_t at) {
			assert(at < __task_queues.size());
			__refs[at]--;
		}
		bool exiting = false; // todo - hide this
		void Reset();
	private:
		std::vector<TaskQueue> __task_queues;
		alignas(OCT_CACHE_LINE_SIZE) std::vector<int> __refs;
	};

	size_t& GetThreadIndex() {
		thread_local size_t index = 0;
		return index;
	}

	class alignas(OCT_CACHE_LINE_SIZE) TaskPools {
	public:
		TaskPool* FetchAdd(size_t& iterative_index) {
			std::shared_lock<std::shared_mutex> lock(__mtx);
			if (iterative_index >= __task_pool_vec.size()) {
				iterative_index = 0;
			}
			if (iterative_index < __task_pool_vec.size()) {
				auto pool = __task_pool_vec[iterative_index++];
				if (!pool->exiting) {
					pool->IncRef(GetThreadIndex());
					return pool;
				}
			}
			return nullptr;
		}
		void InsertPool(TaskPool* pool) {
			assert(pool);
			std::unique_lock<std::shared_mutex> lock(__mtx);
			__task_pool_vec.push_back(pool);
		}
		void RemovePool(TaskPool* pool) {
			std::unique_lock<std::shared_mutex> lock(__mtx);
			for (auto iter = __task_pool_vec.begin(); iter != __task_pool_vec.end(); ++iter) {
				if (*iter == pool) {
					__task_pool_vec.erase(iter);
					return;
				}
			}
		}
	private:
		std::vector<TaskPool*> __task_pool_vec;
		std::shared_mutex __mtx;
	};

	TaskPools task_pools;

	TaskPool::TaskPool() {
		__task_queues.resize(num_thread+1);
		__refs.resize(num_thread+1, 0);
		task_pools.InsertPool(this);
	}

	TaskPool::~TaskPool() {
		exiting = true;
		auto num_ref = __refs.size();
		for (size_t i = 1; i < num_ref; ++i) {
			while (__refs[i]) {
				_mm_pause();
			}
		}
		task_pools.RemovePool(this);
	}

	void TaskPool::Reset() {
		//exiting = true;
		//auto num_ref = __refs.size();
		//for (size_t i = 1; i <= num_ref; ++i) {
		//	while (__refs[i]) {
		//		_mm_pause();
		//	}
		//}
		//__task_queues.resize(num_thread + 1);
		//__refs.resize(num_thread + 1, 0);
		//exiting = false;
	}

	TaskPool* GetTaskPool() {
		if (GetThreadIndex() == 0) {
			thread_local TaskPool main_thread_task_pool;
			if (main_thread_task_pool.Size() != num_thread + 1) {
				main_thread_task_pool.Reset();
			}
			return &main_thread_task_pool;
		}
		else {
			thread_local size_t iterative_index = {};
			return task_pools.FetchAdd(iterative_index);
		}
	}

	void Task::Run() {
		TaskPool* task_pool = GetTaskPool();
		if (__fn && __partitioner && task_pool) {
			auto thread_index = GetThreadIndex();
			std::ptrdiff_t end = __end;
			while ((end = __partitioner->Partition(__begin, __end)) < __end) {
				Task sub_task(__fn, __partitioner, end, __end);
				if (!task_pool->PushAt(std::move(sub_task), thread_index)) {
					break;
				}
				__end = end;
			}
			task_pool->DecRef(thread_index);
		}
		if (__fn) {
			assert(__begin <= __end);
			(*__fn)(__begin, __end);
		}
	}

	void ParallFor(Fn* fn, std::ptrdiff_t begin, std::ptrdiff_t end) {
		if (!fn || begin >= end) {
			return;
		}
		if (GetThreadIndex()) {
			// if it's sub-thread
			Task task(fn, &static_partitioner, begin, end);
			task.Run();
		}
		else {
			// if it's main thread
			alignas(OCT_CACHE_LINE_SIZE) std::atomic<std::ptrdiff_t> counter{ begin };
			Fn wrapper_fn = [&counter, fn](std::ptrdiff_t b, std::ptrdiff_t e) {
				(*fn)(b, e);
				counter.fetch_add(e - b, OCT_ATOM_RLX);
			};

			Task task(&wrapper_fn, &static_partitioner, begin, end);
			task.Run();

			TaskPool* task_pool = GetTaskPool();
			while (task = task_pool->PopAt(0)) {
				task.Run();
			}

			bool done_task = {};
			assert(task_pool);

			while (counter.load(OCT_ATOM_RLX) < end) {
				done_task = false;
				for (size_t i = 1; i <= num_thread; ++i) {
					task = task_pool->PopAt(i);
					if (task) {
						task.Run();
						while (task = task_pool->PopAt(0)) {
							task.Run();
						}
						done_task = true;
					}
				}
				if (!done_task) {
					_mm_pause();
				}
			}
		}
	}

	void ParallFor(Fn* fn, std::ptrdiff_t total) {
		ParallFor(fn, 0, total);
	}

	void ThreadEntry(size_t index) {
		assert(index > 0);
		assert(index <= thread_data_vec.size());

		GetThreadIndex() = index;
		bool done_task = {};
		bool has_task = {};
		TaskPool* task_pool = {};
		ThreadData& thread_data = thread_data_vec[index - 1];
		thread_data.tid = std::this_thread::get_id();

		while (!thread_data.exit) {
			done_task = false;
			task_pool = GetTaskPool();
			if (task_pool) {
				do {
					has_task = false;
					for (size_t i = 0; i <= num_thread; ++i) {
						auto task = task_pool->PopAt(i);
						if (task) {
							task.Run();
							while (task = task_pool->PopAt(index)) {
								task.Run();
							}
							done_task = has_task = true;
						}
					}
				} while (has_task);
				task_pool->DecRef(index);
			}
			if (!done_task) {
				std::this_thread::yield(); // release cpu when idel
			}
		}
	}

	void StartThreadPool(size_t num_thread) {
		assert(octopus::num_thread == 0 && thread_data_vec.empty());
		OCT_ENFORCE(GetThreadIndex() == 0, "StartThreadPool should only be called from main thread");
		octopus::num_thread = num_thread;
		thread_data_vec.resize(num_thread);
		for (size_t index = 1; index <= num_thread; ++index) {
			threads.emplace_back(ThreadEntry, index);
		}
	}

	void StopThreadPool() {
		OCT_ENFORCE(GetThreadIndex() == 0, "StopThreadPool should only be called from main thread");
		std::for_each(thread_data_vec.begin(), thread_data_vec.end(),
			[](ThreadData& thread_data) { thread_data.exit = true; });
		std::for_each(threads.begin(), threads.end(),
			[](std::thread& t) { t.join(); });
		thread_data_vec.clear();
		octopus::num_thread = 0;
	}
}