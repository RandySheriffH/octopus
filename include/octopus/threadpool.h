#pragma once
#include <unordered_map>
#include <shared_mutex>
#include <functional>
#include <algorithm>
#include <cassert>
#include <thread>
#include <vector>
#include <atomic>
#include <mutex>

#ifndef _WIN32
#include <xmmintrin.h>
#endif

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

#define PAD int64_t __pad = 0

namespace octopus {

    using Iter = std::atomic_int32_t;
    using Fn = std::function<void(std::ptrdiff_t, std::ptrdiff_t)>;

    template<typename T, size_t CAPACITY>
    class alignas(OCT_CACHE_LINE_SIZE) Queue {
    public:
        struct alignas(OCT_CACHE_LINE_SIZE) Slot {
            Slot() = default;
            Slot& operator=(const Slot& slot) {
                if (this != &slot) {
                    t = slot.t;
                    state.store(slot.state.load(OCT_ATOM_RLX), OCT_ATOM_RLX);
                }
                return *this;
            }
            T t = {};
            //0: empty, 1: loading, 2: unloading, 3: ready
            alignas(OCT_CACHE_LINE_SIZE) std::atomic_short state{ 0 };
        };
        Queue() = default;
        Queue(const Queue& queue) {
            __head.store(queue.__head.load());
            __tail.store(queue.__tail.load());
            for (size_t i = 0; i < CAPACITY; ++i) {
                __slots[i] = queue.__slots[i];
            }
        };
        T PopHead(bool poll = true) {
            do {
                auto head = __head.load(OCT_ATOM_RLX);
                if (head >= 0 &&
                    __head.compare_exchange_weak(head, -head - 1, OCT_ATOM_RLX, OCT_ATOM_RLX)) {
                    short state = ready;
                    if (__slots[head].state.compare_exchange_weak(state, unloading, OCT_ATOM_ACQ, OCT_ATOM_RLX)) {
                        T t = std::move(__slots[head].t);
                        __slots[head].t = {};
                        __slots[head].state.store(empty, OCT_ATOM_REL);
                        auto next_head = (head + 1) % CAPACITY;
                        __head.store(next_head, OCT_ATOM_RLX);
                        return t;
                    }
                    else {
                        __head.store(head, OCT_ATOM_RLX);
                        return {};
                    }
                }
            } while (poll);
            return {};
        }
        T PopTail(bool poll = true) {
            do {
                auto tail = __tail.load(OCT_ATOM_RLX);
                if (tail >= 0 &&
                    __tail.compare_exchange_weak(tail, -tail - 1, OCT_ATOM_RLX, OCT_ATOM_RLX)) {
                    auto prev_tail = (tail - 1 + CAPACITY) % CAPACITY;
                    short state = ready;
                    if (__slots[prev_tail].state.compare_exchange_weak(state, unloading, OCT_ATOM_RLX, OCT_ATOM_RLX)) {
                        T t = std::move(__slots[prev_tail].t);
                        __slots[prev_tail].t = {};
                        __slots[prev_tail].state.store(empty, OCT_ATOM_RLX);
                        __tail.store(prev_tail, OCT_ATOM_REL);
                        return t;
                    }
                    else {
                        // tail is not ready
                        __tail.store(tail, OCT_ATOM_RLX);
                        return {};
                    }
                }
            } while (poll);
            return {};
        }
        bool PushTail(T&& t) {
            while (true) {
                auto tail = __tail.load(OCT_ATOM_RLX);
                if (tail >= 0 &&
                    __tail.compare_exchange_weak(tail, -tail - 1, OCT_ATOM_RLX, OCT_ATOM_RLX)) {
                    short state = empty;
                    if (__slots[tail].state.compare_exchange_weak(state, loading, OCT_ATOM_RLX, OCT_ATOM_RLX)) {
                        __slots[tail].t = std::move(t);
                        __slots[tail].state.store(ready, OCT_ATOM_REL);
                        auto next_tail = (tail + 1) % CAPACITY;
                        __tail.store(next_tail, OCT_ATOM_RLX);
                        return true;
                    }
                    else {
                        __tail.store(tail, OCT_ATOM_REL);
                        return false;
                    }
                }
            }
        }
    private:
        alignas(OCT_CACHE_LINE_SIZE) Iter __head{};
        alignas(OCT_CACHE_LINE_SIZE) Iter __tail {};
        alignas(OCT_CACHE_LINE_SIZE) Slot __slots[CAPACITY];
        const short empty = 0;
        const short loading = 1;
        const short unloading = 2;
        const short ready = 3;
        PAD;
    };

    template<typename T, size_t CAPACITY>
    class alignas(OCT_CACHE_LINE_SIZE) LinkedList {
    public:
        struct alignas(OCT_CACHE_LINE_SIZE) Node {
            Node(T&& t) : __t(t) {}
            Node(const T& t) : __t(t) {}
            T __t{};
            std::atomic_bool __locked{ false };
            Node* __prev{};
            Node* __next{};
        };
        explicit LinkedList() {
            __head.__next = &__tail;
            __tail.__prev = &__head;
        }
        void Prepend(Node* node) {
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
            while (!node->__locked.compare_exchange_weak(
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
        }
        T Head() {
            bool locked = false;
            while (!__head.__locked.compare_exchange_weak(
                locked, true, OCT_ATOM_RLX, OCT_ATOM_RLX)) {
                locked = false;
                _mm_pause();
            }
            T t = __head.__next->__t;
            __head.__locked.store(false, OCT_ATOM_RLX);
            return t;
        }
    private:
        Node __head{ T{} };
        Node __tail{ T{} };
    };

    struct Partitioner {
        virtual std::ptrdiff_t Partition(std::ptrdiff_t, std::ptrdiff_t) const = 0;
    };

    class BinaryPartitioner : public Partitioner {
    public:
        BinaryPartitioner(std::ptrdiff_t min_chunk) : __min_chunk(min_chunk) {
        }
        std::ptrdiff_t Partition(std::ptrdiff_t begin, std::ptrdiff_t end) const override {
            auto half = (end - begin) >> 1;
            return half < __min_chunk ? end : begin + half;
        }
    private:
        std::ptrdiff_t __min_chunk;
    };

    class StaticPartitioner : public Partitioner {
    public:
        StaticPartitioner(std::ptrdiff_t chuck_size) :
            __chuck_size(chuck_size) {
            assert(__chuck_size);
        }
        std::ptrdiff_t Partition(std::ptrdiff_t begin, std::ptrdiff_t end) const override {
            return std::min(end, begin + __chuck_size);
        }
    private:
        const std::ptrdiff_t __chuck_size;
    };

    BinaryPartitioner s_binary_partitioner(1);

    class AffinityPartitioner : public Partitioner {
    public:
        AffinityPartitioner(std::ptrdiff_t num_chunk, std::ptrdiff_t min_chunk_size) :
            __num_chuck(num_chunk), __min_chuck_size(min_chunk_size) {
            assert(__num_chuck > 0);
            assert(__min_chuck_size > 0);
        }
        std::ptrdiff_t Partition(std::ptrdiff_t begin, std::ptrdiff_t end) const override {
            auto total = end - begin;
            if (total <= __min_chuck_size) {
                return end;
            }
            else {
                auto avg_chuck_size = total / __num_chuck;
                if (avg_chuck_size * __num_chuck < total) {
                    ++avg_chuck_size;
                }
                auto mid = end - avg_chuck_size;
                if (mid - begin < __min_chuck_size) {
                    return end;
                }
                else {
                    return mid;
                }
            }
        }
    private:
        const std::ptrdiff_t __num_chuck;
        const std::ptrdiff_t __min_chuck_size;
    };

	class alignas(OCT_CACHE_LINE_SIZE) Task {
	public:
		Task() = default;
		Task(const Fn& fn,
			std::ptrdiff_t begin,
			std::ptrdiff_t end) :
			__fn(fn), __begin(begin), __end(end) {}

		Task(const Task& task) = default;
		Task(Task&& task) = default;

		operator bool() const { return __fn.operator bool(); }
		Task& operator = (const Task& task) = default;
		Task& operator = (Task&& task) = default;

		void Run();

	private:
		Fn __fn;
		std::ptrdiff_t __begin = {};
		std::ptrdiff_t __end = {};
		PAD;
	};

    // using TaskQueue = Queue<Task, 128>;
    using TaskQueue = Queue<Task, 4>;

    // each main thread owns a task pool
    class alignas(OCT_CACHE_LINE_SIZE) TaskPool {
    public:
        TaskPool(size_t num_queue);
        ~TaskPool() = default;
        Task PopHeadAt(size_t at, bool poll) {
            assert(at < __task_queues.size());
            return __task_queues[at].PopHead(poll);
        }
        Task PopTailAt(size_t at, bool poll) {
            assert(at < __task_queues.size());
            return __task_queues[at].PopTail(poll);
        }
        bool PushTailAt(Task&& t, size_t at) {
            assert(at < __task_queues.size());
            return __task_queues[at].PushTail(std::forward<Task>(t));
        }
        size_t Size() const {
            return __task_queues.size();
        }
        bool exiting = false; // todo - hide this
        void Reset();
        const Partitioner* partitioner = {};
    private:
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
            __thread_datas.resize(__num_thread - 1);
            __threads.reserve(__num_thread - 1);
            for (size_t index = 1; index < __num_thread; ++index) {
                __threads.emplace_back(&ThreadPool::ThreadEntry, this, index);
            }
        }

        ~ThreadPool() {
            std::for_each(__thread_datas.begin(), __thread_datas.end(),
                [](ThreadData& thread_data) { thread_data.exit = true; });
            NotifyAll();
            std::for_each(__threads.begin(), __threads.end(),
                [this](std::thread& t) { t.join(); });
            __thread_datas.clear();
        }

        void ParallFor(const Fn& fn, std::ptrdiff_t total, const Partitioner& partitioner = s_binary_partitioner) {
            ParallFor(fn, 0, total, partitioner);
        }

        void ParallFor(const Fn& fn, std::ptrdiff_t begin, std::ptrdiff_t end, const Partitioner& partitioner = s_binary_partitioner) {
            if (!fn || begin >= end) {
                return;
            }

            if (GetThreadIndex()) {
                // if it's sub-thread
                Task task(fn, begin, end);
                task.Run();
            }
            else {
                // if it's main thread
                *GetThreadPool() = this;
                auto task_pool = *GetTaskPool();
                task_pool->partitioner = &partitioner;

                LinkedList<TaskPool*, 32>::Node list_node(task_pool);
                __task_pools.Prepend(&list_node);

                alignas(OCT_CACHE_LINE_SIZE) std::atomic<std::ptrdiff_t> counter{ begin };
                Fn wrapper_fn = [&counter, fn](std::ptrdiff_t b, std::ptrdiff_t e) {
                    fn(b, e);
                    counter.fetch_add(e - b, OCT_ATOM_RLX);
                    };

                Task task(wrapper_fn, begin, end);
                task.Run();

                assert(task_pool);
                while (task = task_pool->PopTailAt(0, true)) {
                    task.Run();
                }

                bool done_tasks = false;
                while (counter.load(OCT_ATOM_RLX) < end) {
                    done_tasks = false;
                    for (size_t i = 1; i < __num_thread;) {
                        task = task_pool->PopHeadAt(i, false);
                        if (task) {
                            task.Run();
                            while (task = task_pool->PopTailAt(0, true)) {
                                task.Run();
                            }
                            done_tasks = true;
                        }
                        else {
                            ++i;
                        }
                    }
                    if (!done_tasks) {
                        _mm_pause(); // see if yield better the perf
                    }
                }

                assert(counter.load(OCT_ATOM_RLX) == end);
                __task_pools.Remove(&list_node);
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

            TaskPool* task_pool = {};
            ThreadData& thread_data = __thread_datas[index - 1];
            thread_data.tid = std::this_thread::get_id();

            constexpr size_t num_spin = 10;
            while (!thread_data.exit) {
                *GetTaskPool() = {};
                task_pool = {};
                for (size_t i = 0; i < num_spin; ++i) {
                    task_pool = __task_pools.Head();
                    if (task_pool) {
                        break;
                    }
                }
                if (task_pool) {
                    *GetTaskPool() = task_pool;
                    for (size_t j = 0; j < __num_thread;) {
                        auto task = task_pool->PopHeadAt(j, false);
                        if (task) {
                            task.Run();
                            while (task = task_pool->PopTailAt(index, true)) {
                                task.Run();
                            }
                        }
                        else {
                          ++j;
                        }
                    }
                }
                else {
                    WaitForTask();
                }
            }
        }

        void NotifyAll() {
            __cv.notify_all();
        }

        void WaitForTask() {
            std::unique_lock<std::mutex> lock(__mtx);
            __cv.wait_for(lock, std::chrono::milliseconds{ 300 });
        }

        const size_t __num_thread; // including main thread
        std::vector<ThreadData> __thread_datas;
        std::vector<std::thread> __threads;
        LinkedList<TaskPool*, 32> __task_pools;
        std::condition_variable __cv;
        std::mutex __mtx;
    };

    TaskPool::TaskPool(size_t num_queue) {
        __task_queues.resize(num_queue);
    }

    void Task::Run() {
        TaskPool** task_pool = ThreadPool::GetTaskPool();
        const Partitioner* partitioner = (*task_pool)->partitioner;
        assert(partitioner);
        if (__fn && *task_pool) {
            auto thread_index = ThreadPool::GetThreadIndex();
            std::ptrdiff_t mid = __end;
            while ((mid = partitioner->Partition(__begin, __end)) < __end) {
                Task sub_task(__fn, mid, __end);
                if ((*task_pool)->PushTailAt(std::move(sub_task), thread_index)) {
                    (*ThreadPool::GetThreadPool())->NotifyAll();
                    __end = mid;
                }
                else {
                    break;
                }
            }
        }
        if (__fn) {
            assert(__begin <= __end);
            __fn(__begin, __end);
        }
    }
}