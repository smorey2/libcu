#include <string.h>
#include <pwdcu.h>

__device__ __managed__ struct passwd *m_getpwnam_rc;
__global__ void g_getpwnam(char *name) {
	m_getpwnam_rc = getpwnam(name);
}
struct passwd *dchown_getpwnam_(char *str) {
	size_t strLength = strlen(str) + 1;
	char *d_str;
	cudaMalloc(&d_str, strLength);
	cudaMemcpy(d_str, str, strLength, cudaMemcpyHostToDevice);
	g_getpwnam<<<1, 1>>>(d_str);
	cudaFree(d_str);
	return m_getpwnam_rc;
}
