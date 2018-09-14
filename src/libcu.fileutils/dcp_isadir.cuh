#include <unistdcu.h>
#include <string.h>

// Return TRUE if a filename is a directory. Nonexistant files return FALSE.
__device__ __managed__ bool m_isadir_rc;
__global__ void g_isadir(char *name) {
	struct stat statbuf;
	if (stat(name, &statbuf) < 0) {
		m_isadir_rc = false;
		return;
	}
	m_isadir_rc = S_ISDIR(statbuf.st_mode);
	return;
}

bool dcp_isadir_(char *str) {
	size_t strLength = strlen(str) + 1;
	char *d_str;
	cudaMalloc(&d_str, strLength);
	cudaMemcpy(d_str, str, strLength, cudaMemcpyHostToDevice);
	g_isadir<<<1, 1>>>(d_str);
	cudaFree(d_str);
	return m_isadir_rc;
}
