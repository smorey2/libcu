#include <stdlibcu.h>
#include <ext/mutex.h>

/* Mutex with exponential back-off. */
__host_device__ void mutexSpinLock(void **cancelToken, volatile long *mutex, long cmp, long val, long mask, bool(*func)(void **), void **funcTag, unsigned int msmin, unsigned int msmax) {
	long v; unsigned int ms = msmin;
#if __CUDA_ARCH__
	while ((!cancelToken || *cancelToken) && (v = atomicCAS((int *)mutex, cmp, val)) != cmp) {
		if (v & mask) { if (!func(funcTag)) return; continue; }
#if __CUDA_ARCH__ >= 700
		__nanosleep(ms * 1000);
#else
		usleep(ms * 1000);
#endif
#elif __OS_WIN
	while ((!cancelToken || *cancelToken) && (v = _InterlockedCompareExchange((volatile long *)mutex, cmp, val)) != cmp) {
		if (v & mask) { if (!func(funcTag)) return; continue; }
		Sleep(ms);
#elif __OS_UNIX
	while ((!cancelToken || *cancelToken) && (v = __sync_val_compare_and_swap((long *)mutex, cmp, val)) != cmp) {
		if (v & mask) { if (!func(funcTag)) return; continue; }
		sleep(ms);
#endif
		if (ms < msmax) ms *= 2;
	}
}

/* Mutex set. */
__host_device__ void mutexSet(volatile long *mutex, long val, unsigned int mspause) {
	unsigned int ms = mspause;
#if __CUDA_ARCH__
	atomicExch((int *)mutex, val);
#if __CUDA_ARCH__ >= 700
	if (ms) __nanosleep(ms * 1000);
#else
	if (ms) usleep(ms * 1000);
#endif
#elif __OS_WIN
	_InterlockedExchange((volatile long *)mutex, val);
	if (ms) Sleep(ms);
#elif __OS_UNIX
	__sync_lock_test_and_set((long *)control, val);
	if (ms) sleep(ms);
#endif
}
