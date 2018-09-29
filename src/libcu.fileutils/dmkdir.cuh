#include <ext/pipeline.h>
#include <sys/statcu.h>

__device__ int d_dmkdir_rc;
__global__ void g_dmkdir(pipelineRedir redir, char *str, unsigned short mode) {
	fprintf(redir.out, "TEST\n");
	d_dmkdir_rc = mkdir(str, mode);
}
int dmkdir(pipelineRedir redir, char *str, unsigned short mode) {
	redir.Open();
	size_t strLength = strlen(str) + 1;
	char *d_str;
	cudaMalloc(&d_str, strLength);
	cudaMemcpy(d_str, str, strLength, cudaMemcpyHostToDevice);
	g_dmkdir<<<1, 1>>>(redir, d_str, mode);
	cudaFree(d_str);
	redir.Close();
	int rc; cudaMemcpyFromSymbol(&rc, d_dmkdir_rc, sizeof(rc), 0, cudaMemcpyDeviceToHost); return rc;
}
