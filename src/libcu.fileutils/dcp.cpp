#define _CRT_SECURE_NO_WARNINGS
#include <stdlib.h>
#include <stdio.h>
#include "sentinel-fileutilsmsg.h"
#include <sentinel-client.cpp>
#include <ext/pipeline.cpp>

__forceinline__ bool dcp_isadir_(pipelineRedir redir, char *str) { fileutils_isadir msg(redir, str); return msg.RC; }
__forceinline__ int dcp_(pipelineRedir redir, char *str, char *str2, bool setModes) { fileutils_dcp msg(redir, str, str2, setModes); return msg.RC; }

// Build a path name from the specified directory name and file name. If the directory name is NULL, then the original filename is returned.
// The built path is in a static area, and is overwritten for each call.
char *buildName(char *dirName, char *fileName) {
	if (!dirName || *dirName == '\0')
		return fileName;
	char *cp = strrchr(fileName, '/');
	if (cp)
		fileName = cp + 1;
	static char buf[FILENAME_MAX];
	strcpy(buf, dirName);
	strcat(buf, "/");
	strcat(buf, fileName);
	return buf;
}

int main(int argc, char	**argv) {
	atexit(sentinelClientShutdown);
	sentinelClientInitialize();
	FDTYPE hostRedir[3]; pipelineRedir clientRedir = sentinelClientRedir(hostRedir);
	char *lastArg = argv[argc - 1];
	bool dirflag = dcp_isadir_(clientRedir, lastArg);
	if (argc > 3 && !dirflag) {
		fprintf(stderr, "%s: not a directory\n", lastArg);
		exit(1);
	}
	while (argc-- > 2) {
		char *srcName = argv[1];
		char *destName = lastArg;
		if (dirflag)
			destName = buildName(destName, srcName);
		dcp_(clientRedir, *++argv, destName, false);
	}
	exit(0);
}
