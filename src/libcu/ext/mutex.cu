#include <stdlibcu.h>
#include <ext/mutex.h>

#if __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#define SLEEP(MS) __nanosleep(MS * 1000)
#else
#define SLEEP(MS) usleep(MS * 1000);
#endif
#elif __OS_WIN
#define SLEEP(MS) Sleep(MS)
#elif __OS_UNIX
#define SLEEP(MS) sleep(MS)
#endif

/* Mutex with exponential back-off. */
__host_device__ void mutexSpinLock(void **cancelToken, volatile long *mutex, long cmp, long val, char pred, long predVal, bool(*func)(void **), void **funcTag, unsigned int msmin, unsigned int msmax) {
	long v; unsigned int ms = msmin;
#if __CUDA_ARCH__
	while ((!cancelToken || *cancelToken) && (v = atomicCAS((int *)mutex, cmp, val)) != cmp) {
#elif __OS_WIN
	while ((!cancelToken || *cancelToken) && (v = _InterlockedCompareExchange((volatile long *)mutex, cmp, val)) != cmp) {
#elif __OS_UNIX
	while ((!cancelToken || *cancelToken) && (v = __sync_val_compare_and_swap((long *)mutex, cmp, val)) != cmp) {
#endif
		bool condition = false;
		switch (pred) {
		case MUTEXPRED_AND: condition = v & predVal; break;
		case MUTEXPRED_ANE: condition = (v & predVal) == predVal; break;
		case MUTEXPRED_LTE: condition = v <= predVal; break;
		case MUTEXPRED_GTE: condition = v >= predVal; break;
		}
		if (condition) { if (!func || !func(funcTag)) return; continue; }
		SLEEP(ms);
		if (ms < msmax) ms *= 1.5;
	}
}

/* Mutex set. */
__host_device__ void mutexSet(volatile long *mutex, long val, unsigned int mspause) {
#if __CUDA_ARCH__
	atomicExch((int *)mutex, val);
#elif __OS_WIN
	_InterlockedExchange((volatile long *)mutex, val);
#elif __OS_UNIX
	__sync_lock_test_and_set((long *)mutex, val);
#endif
	if (mspause) SLEEP(mspause);
}
