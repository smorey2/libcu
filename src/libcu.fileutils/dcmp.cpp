#define _CRT_SECURE_NO_WARNINGS
#include <stdlib.h>
#include <stdio.h>
#include "sentinel-fileutilsmsg.h"
#include <sentinel-client.cpp>
#include <ext/pipeline.cpp>

__forceinline__ int dcmp_(pipelineRedir redir, char *str, char *str2) { fileutils_dcmp msg(redir, str, str2); return msg.RC; }

int main(int argc, char	**argv) {
	atexit(sentinelClientShutdown);
	sentinelClientInitialize();
	FDTYPE hostRedir[3]; pipelineRedir clientRedir = sentinelClientRedir(hostRedir);
	int r = dcmp_(clientRedir, argv[1], argv[2]);
	exit(r);
}
