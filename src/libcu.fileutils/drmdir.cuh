#include <ext/pipeline.h>
#include <unistdcu.h>

__device__ int d_drmdir_rc;
__global__ void g_drmdir(pipelineRedir redir, char *str) {
	d_drmdir_rc = rmdir(str);
}
int drmdir(pipelineRedir redir, char *str) {
	redir.Open();
	size_t strLength = strlen(str) + 1;
	char *d_str;
	cudaMalloc(&d_str, strLength);
	cudaMemcpy(d_str, str, strLength, cudaMemcpyHostToDevice);
	g_drmdir<<<1, 1>>>(redir, d_str);
	cudaFree(d_str);
	redir.Close();
	int rc; cudaMemcpyFromSymbol(&rc, d_drmdir_rc, sizeof(rc), 0, cudaMemcpyDeviceToHost); return rc;
}
