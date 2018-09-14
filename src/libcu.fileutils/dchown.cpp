#define _CRT_SECURE_NO_WARNINGS
#include <stdlib.h>
#include <stdio.h>
#include "sentinel-fileutilsmsg.h"
#include <sentinel-client.cpp>
#include <pwdcu.h>

#define	isdecimal(ch) ((ch) >= '0' && (ch) <= '9')
__forceinline__ struct passwd *dchgrp_getpwnam_(char *str) { fileutils_getpwnam msg(str); return msg.RC; }
__forceinline__ int dchown_(char *str, int uid) { fileutils_dchown msg(str, uid); return msg.RC; }

int main(int argc, char	**argv) {
	atexit(sentinelClientShutdown);
	sentinelClientInitialize();
	char *cp = argv[1];
	int uid;
	if (isdecimal(*cp)) {
		uid = 0;
		while (isdecimal(*cp))
			uid = uid * 10 + (*cp++ - '0');
		if (*cp) {
			fprintf(stderr, "Bad uid value\n");
			exit(1);
		}
	}
	else {
		struct passwd *pwd = dchgrp_getpwnam_(cp);
		if (!pwd) {
			fprintf(stderr, "Unknown user name\n");
			exit(1);
		}
		uid = pwd->pw_uid;
	}
	//
	argc--;
	argv++;
	while (argc-- > 1) {
		argv++;
		if (dchown_(*argv, uid))
			perror(*argv);
	}
	exit(0);
}
