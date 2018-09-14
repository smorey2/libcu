#define _CRT_SECURE_NO_WARNINGS
#include <stdlib.h>
#include <stdio.h>
#include "sentinel-fileutilsmsg.h"
#include <sentinel-client.cpp>
#include <grpcu.h>

#define	isdecimal(ch) ((ch) >= '0' && (ch) <= '9')
__forceinline__ struct group *dchgrp_getgrnam_(char *str) { fileutils_getgrnam msg(str); return msg.RC; }
__forceinline__ int dchgrp_(char *str, int gid) { fileutils_dchgrp msg(str, gid); return msg.RC; }

int main(int argc, char **argv) {
	atexit(sentinelClientShutdown);
	sentinelClientInitialize();
	char *cp = argv[1];
	int gid;
	if (isdecimal(*cp)) {
		gid = 0;
		while (isdecimal(*cp))
			gid = gid * 10 + (*cp++ - '0');
		if (*cp) {
			fprintf(stderr, "Bad gid value\n");
			exit(1);
		}
	}
	else {
		struct group *grp = dchgrp_getgrnam_(cp);
		if (!grp) {
			fprintf(stderr, "Unknown group name\n");
			exit(1);
		}
		gid = grp->gr_gid;
	}
	//
	argc--;
	argv++;
	while (argc-- > 1) {
		argv++;
		if (dchgrp_(*argv, gid))
			perror(*argv);
	}
	exit(0);
}
