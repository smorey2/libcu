#include <ext/pipeline.h>
#include <sys/statcu.h>
#include <unistdcu.h>

__device__ int d_dchown_rc;
__global__ void g_dchown(pipelineRedir redir, char *str, int uid) {
	struct stat	statbuf;
	d_dchown_rc = (stat(str, &statbuf) < 0 || chown(str, uid, statbuf.st_gid) < 0);
}
int dchown(pipelineRedir redir, char *str, int uid) {
	redir.Open();
	size_t strLength = strlen(str) + 1;
	char *d_str;
	cudaMalloc(&d_str, strLength);
	cudaMemcpy(d_str, str, strLength, cudaMemcpyHostToDevice);
	g_dchown<<<1, 1>>>(redir, d_str, uid);
	cudaFree(d_str);
	redir.Close();
	int rc; cudaMemcpyFromSymbol(&rc, d_dchown_rc, sizeof(rc), 0, cudaMemcpyDeviceToHost); return rc;
}
