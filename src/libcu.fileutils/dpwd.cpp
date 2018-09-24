#define _CRT_SECURE_NO_WARNINGS
#include <stdlib.h>
#include <stdio.h>
#include "sentinel-fileutilsmsg.h"
#include <sentinel-client.cpp>
#include <ext/pipeline.cpp>

__forceinline__ int dpwd_(pipelineRedir redir, char *ptr) { fileutils_dpwd msg(redir); strcpy(ptr, msg.Ptr); return msg.RC; }

int main(int argc, char **argv) {
	atexit(sentinelClientShutdown);
	sentinelClientInitialize();
	FDTYPE hostRedir[3]; pipelineRedir clientRedir = sentinelClientRedir(hostRedir);
	char pwd[FILENAME_MAX];
	if (dpwd_(clientRedir, pwd)) {
		fprintf(stderr, "pwd: cannot get current directory\n");
		exit(1);
	}
	printf("%s\n", pwd);
	exit(0);
}
