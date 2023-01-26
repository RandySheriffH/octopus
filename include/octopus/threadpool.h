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
	using Fn = std::function<void(std::ptrdiff_t, std::ptrdiff_t)>;

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
					if (__head.compare_exchange_weak(head, - head - 1, OCT_ATOM_ACQ, OCT_ATOM_RLX)) {
						T t = __slots[head];
						__slots[head] = {};
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
					if (__tail.compare_exchange_weak(tail, - tail - 1, OCT_ATOM_RLX, OCT_ATOM_RLX)) {
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

	template<typename T, size_t CAPACITY>
	class alignas(OCT_CACHE_LINE_SIZE) LinkedList {
	public:
		struct alignas(OCT_CACHE_LINE_SIZE) Node {
			Node(T&& t) : __t(t) {}
			T __t{};
			std::atomic_bool __locked{ false };
			Node* __prev{};
			Node* __next{};
		};
		explicit LinkedList() {
			__head.__next = &__tail;
			__tail.__prev = &__head;
		}
		Node* Prepend(T&& t) {
			Node* node = new Node(std::move(t));
			bool locked = false;
			while (!__head.__locked.compare_exchange_weak(
				locked, true, OCT_ATOM_RLX, OCT_ATOM_RLX)) {
				locked = false;
				_mm_pause();
			}
			while (!__head.__next->__locked.compare_exchange_weak(
				locked, true, OCT_ATOM_RLX, OCT_ATOM_RLX)) {
				locked = false;
				_mm_pause();
			}
			Node* next = __head.__next;
			__head.__next = node;
			node->__next = next;
			node->__prev = &__head;
			next->__prev = node;
			__head.__locked.store(false, OCT_ATOM_RLX);
			next->__locked.store(false, OCT_ATOM_RLX);
			return node;
		}
		void Remove(Node* node) {
			assert(node);
			assert(node != &__head);
			assert(node != &__tail);
			bool locked = false;
			while (!node->__prev->__locked.compare_exchange_weak(
				locked, true, OCT_ATOM_RLX, OCT_ATOM_RLX)) {
				locked = false;
				_mm_pause();
			}
			while (!node->__next->__locked.compare_exchange_weak(
				locked, true, OCT_ATOM_RLX, OCT_ATOM_RLX)) {
				locked = false;
				_mm_pause();
			}
			Node* prev = node->__prev;
			Node* next = node->__next;
			prev->__next = next;
			next->__prev = prev;
			prev->__locked.store(false, OCT_ATOM_RLX);
			next->__locked.store(false, OCT_ATOM_RLX);
			delete node;
		}
		T Head() const {
			bool locked = false;
			while (!__head.__next->__locked.compare_exchange_weak(
				locked, true, OCT_ATOM_RLX, OCT_ATOM_RLX)) {
				locked = false;
				_mm_pause();
			}
			T t = __head.__next->__t;
			__head.__next->__locked.store(false, OCT_ATOM_RLX);
			return t;
		}
	private:
		Node __head{ T{} };
		Node __tail{ T{} };
	};

	struct Partitioner {
		virtual std::ptrdiff_t Partition(std::ptrdiff_t, std::ptrdiff_t) const = 0;
	};

	class StaticPartitioner : public Partitioner {
	public:
		StaticPartitioner(std::ptrdiff_t chuck_size):
			__chuck_size(chuck_size) {
			assert(__chuck_size);
		}
		std::ptrdiff_t Partition(std::ptrdiff_t begin, std::ptrdiff_t end) const override {
			return std::min(end, begin + __chuck_size);
		}
	private:
		const std::ptrdiff_t __chuck_size;
	};

	StaticPartitioner default_partitioner(1);

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

	using TaskQueue = Queue<Task, 64>;

	// each main thread owns a task pool
	class alignas(OCT_CACHE_LINE_SIZE) TaskPool {
	public:
		TaskPool(size_t num_thread);
		~TaskPool() = default;
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
		bool exiting = false; // todo - hide this
		void Reset();
	private:
		const size_t __num_thread;
		std::vector<TaskQueue> __task_queues;
	};

	struct alignas(OCT_CACHE_LINE_SIZE) ThreadData {
		bool exit = false;
		std::thread::id tid = {};
	};

	class ThreadPool {
	public:
		friend TaskPool::TaskPool(size_t num_thread);
		friend TaskPool::~TaskPool();
		friend void Task::Run();

		//num_thread include main thread, so only "num_thread - 1" threads will be created
		ThreadPool(size_t num_thread) : __num_thread(num_thread) {
			__thread_datas.resize(__num_thread-1);
			for (size_t index = 1; index < __num_thread; ++index) {
				__threads.emplace_back(&ThreadPool::ThreadEntry, this, index);
			}
		}

		~ThreadPool() {
			std::for_each(__thread_datas.begin(), __thread_datas.end(),
				[](ThreadData& thread_data) { thread_data.exit = true; });
			std::for_each(__threads.begin(), __threads.end(),
				[](std::thread& t) { t.join(); });
			__thread_datas.clear();
		}

		void ParallFor(Fn* fn, std::ptrdiff_t total, Partitioner* partitioner = &default_partitioner) {
			ParallFor(fn, 0, total, partitioner);
		}

		void ParallFor(Fn* fn, std::ptrdiff_t begin, std::ptrdiff_t end, Partitioner* partitioner = &default_partitioner) {
			if (!fn || begin >= end) {
				return;
			}

			if (GetThreadIndex()) {
				// if it's sub-thread
				Task task(fn, partitioner, begin, end);
				task.Run();
			}
			else {
				*GetThreadPool() = this;
				auto* node = __task_pools.Prepend(std::move(*GetTaskPool()));
				// if it's main thread
				alignas(OCT_CACHE_LINE_SIZE) std::atomic<std::ptrdiff_t> counter{ begin };
				Fn wrapper_fn = [&counter, fn](std::ptrdiff_t b, std::ptrdiff_t e) {
					(*fn)(b, e);
					counter.fetch_add(e - b, OCT_ATOM_RLX);
				};

				Task task(&wrapper_fn, partitioner, begin, end);
				task.Run();

				TaskPool* task_pool = *GetTaskPool();
				while (task = task_pool->PopAt(0)) {
					task.Run();
				}

				bool done_task = false;
				assert(task_pool);

				while (counter.load(OCT_ATOM_RLX) < end) {
					done_task = false;
					for (size_t i = 1; i < __num_thread; ++i) {
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

				__task_pools.Remove(node);
			}
		}

		size_t NumThread() const {
			return __num_thread;
		}

	private:

		static size_t& GetThreadIndex() {
			thread_local size_t index = 0;
			return index;
		}

		static ThreadPool** GetThreadPool() {
			thread_local ThreadPool* thread_pool = nullptr;
			return &thread_pool;
		}

		static TaskPool** GetTaskPool() {
			ThreadPool* tp = *GetThreadPool();
			assert(tp);
			if (GetThreadIndex()) {
				thread_local TaskPool* main_pool_ptr = {};
				return &main_pool_ptr;
			}
			else {
				thread_local TaskPool main_pool(tp->NumThread());
				thread_local TaskPool* main_pool_ptr = &main_pool;
				return &main_pool_ptr;
			}
		}

		void ThreadEntry(size_t index) {
			assert(index > 0);
			assert(index - 1 < __thread_datas.size());

			GetThreadIndex() = index;
			*GetThreadPool() = this;

			bool done_task = {};
			bool has_task = {};
			TaskPool* task_pool = {};
			ThreadData& thread_data = __thread_datas[index - 1];
			thread_data.tid = std::this_thread::get_id();

			while (!thread_data.exit) {
				done_task = false;
				task_pool = __task_pools.Head();
				if (task_pool) {
					*GetTaskPool() = task_pool;
					do {
						has_task = false;
						for (size_t i = 0; i < __num_thread; ++i) {
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
				}
				if (!done_task) {
					std::this_thread::yield(); // release cpu when idel
				}
			}
		}

		const size_t __num_thread; // including main thread
		std::vector<ThreadData> __thread_datas;
		std::vector<std::thread> __threads;
		LinkedList<TaskPool*,32> __task_pools;
	};

	TaskPool::TaskPool(size_t num_thread) : __num_thread(num_thread) {
		__task_queues.resize(num_thread);
	}

	void Task::Run() {
		TaskPool** task_pool = ThreadPool::GetTaskPool();
		if (__fn && __partitioner && *task_pool) {
			auto thread_index = ThreadPool::GetThreadIndex();
			std::ptrdiff_t end = __end;
			while ((end = __partitioner->Partition(__begin, __end)) < __end) {
				Task sub_task(__fn, __partitioner, end, __end);
				if (!(*task_pool)->PushAt(std::move(sub_task), thread_index)) {
					break;
				}
				__end = end;
			}
		}
		if (__fn) {
			assert(__begin <= __end);
			(*__fn)(__begin, __end);
		}
	}
}