#include <stdlibcu.h>
#include <ext/mutex.h>

/* Mutex with exponential back-off. */
__device__ void mutex_lock(unsigned int *mutex) {
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
__device__ void mutex_unlock(unsigned int *mutex) {
	atomicExch(mutex, 0);
}