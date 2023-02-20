octopus is a work in progress, expect more details to come.
A few implementations to be reflected:
1. the lock-free deque;
2. the lock-free linked list;
A few things to be explored for a single numa node:
1. How to avoid threads concurrently accessing the same counter for task completion?
   So far an intuitive idea is to have hierachical counters - so each thread only access a counter
   of its own thread group, the main thread track all such counters and wait;
2. What are best ways to choose victims for task-stealing? if still taking the hierachical approach,
   threads of same group could do it randomly, or sequentially (a thread steal from its prev and next neighbor).
   between groups a leader thread will be assigned the privilege of stealing, and further between sockets another
   leader thread will do the job...
3. Which set of tests should be selected for benchmarking? qsort, merge sort?