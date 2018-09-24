#define _CRT_SECURE_NO_WARNINGS
#include <stdlib.h>
#include <stdio.h>
#include "sentinel-fileutilsmsg.h"
#include <sentinel-client.cpp>
#include <ext/pipeline.cpp>

__forceinline__ int dpwd_(pipelineRedir redir, char *ptr) { fileutils_dpwd msg(redir); strcpy(ptr, msg.Ptr); return msg.RC; }
__forceinline__ int dcd_(pipelineRedir redir, char *str) { fileutils_dcd msg(redir, str); return msg.RC; }

int main(int argc, const char **argv) {
	atexit(sentinelClientShutdown);
	sentinelClientInitialize();
	FDTYPE hostRedir[3]; pipelineRedir clientRedir = sentinelClientRedir(hostRedir);
	if (argc <= 1 || argc > 2) {
		char pwd[FILENAME_MAX];
		if (!dpwd_(clientRedir, pwd)) {
			printf("%s\n", pwd);
			exit(1);
		}
	}
	int r = dcd_(clientRedir, (char *)argv[1]);
	if (!r) {
		fprintf(stderr, "%s: %s: %s\n", argv[0], argv[1], strerror(r));
		exit(0);
	}
	exit(0);
}
