#include <ext/pipeline.h>
#include <string.h>
#include <grpcu.h>

__device__ __managed__ struct group *m_getgrnam_rc;
__global__ void g_getgrnam(pipelineRedir redir, char *name) {
	m_getgrnam_rc = getgrnam(name);
}
struct group *dchgrp_getgrnam(pipelineRedir redir, char *str) {
	redir.Open();
	size_t strLength = strlen(str) + 1;
	char *d_str;
	cudaMalloc(&d_str, strLength);
	cudaMemcpy(d_str, str, strLength, cudaMemcpyHostToDevice);
	g_getgrnam<<<1, 1>>>(redir, d_str);
	cudaFree(d_str);
	redir.Close();
	return m_getgrnam_rc;
}
