#include <string.h>
#include <grpcu.h>

__device__ __managed__ struct group *m_getgrnam_rc;
__global__ void g_getgrnam(char *name) {
	m_getgrnam_rc = getgrnam(name);
}
struct group *dchgrp_getgrnam(char *str) {
	size_t strLength = strlen(str) + 1;
	char *d_str;
	cudaMalloc(&d_str, strLength);
	cudaMemcpy(d_str, str, strLength, cudaMemcpyHostToDevice);
	g_getgrnam<<<1, 1>>>(d_str);
	cudaFree(d_str);
	return m_getgrnam_rc;
}
