#include <stdlibcu.h>
#include <ext/mutex.h>

/* Mutex with exponential back-off. */
__device__ void mutexLock(unsigned int *mutex) {
#if __CUDA_ARCH__ >= 700
	unsigned int ns = 8;
#endif
	while (atomicCAS(mutex, 0, 1) == 1) {
#if __CUDA_ARCH__ >= 700
		__nanosleep(ns);
		if (ns < 256)
			ns *= 2;
#endif
	}
}

/* Mutex unlock */
__device__ void mutexUnlock(unsigned int *mutex) {
	atomicExch(mutex, 0);
}

/* Mutex held */
__device__ int mutexHeld(unsigned int *mutex) {
	return *mutex == 1;
}
