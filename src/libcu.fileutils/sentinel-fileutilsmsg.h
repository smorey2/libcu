/*
sentinel-fileutilsmsg.h - messages for sentinel
The MIT License

Copyright (c) 2016 Sky Morey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

#ifndef _SENTINEL_FILEUTILSMSG_H
#define _SENTINEL_FILEUTILSMSG_H
#define HAS_GPU 0
#define HAS_HOSTSENTINEL 1
#include <sentinel.h>
#include <string.h>

enum {
	FILEUTILS_DCAT = 10,
	FILEUTILS_DCHGRP,
	FILEUTILS_GETGRNAM, // DCHGRP
	FILEUTILS_DCHMOD,
	FILEUTILS_DCHOWN,
	FILEUTILS_GETPWNAM, // DCHOWN
	FILEUTILS_DCMP,
	FILEUTILS_DCP,
	FILEUTILS_ISADIR, // DCP
	FILEUTILS_DGREP,
	FILEUTILS_DLS,
	FILEUTILS_DMKDIR,
	FILEUTILS_DMORE,
	FILEUTILS_DMV,
	FILEUTILS_DRM,
	FILEUTILS_DRMDIR,
	FILEUTILS_DPWD,
	FILEUTILS_DCD,
};

struct fileutils_dcat {
	sentinelClientMessage Base;
	char *Str;
	fileutils_dcat(pipelineRedir redir, char *str) : Base(redir, FILEUTILS_DCAT, SENTINELFLOW_WAIT, SENTINEL_CHUNK), Str(str) { sentinelClientSend(&Base.base, sizeof(fileutils_dcat), PtrsIn); }
	int RC;
	sentinelInPtr PtrsIn[2] = {
		{ &Str, -1 },
		nullptr
	};
};

struct fileutils_dchgrp {
	sentinelClientMessage Base;
	char *Str; int Gid;
	fileutils_dchgrp(pipelineRedir redir, char *str, int gid) : Base(redir, FILEUTILS_DCHGRP, SENTINELFLOW_WAIT, SENTINEL_CHUNK), Str(str), Gid(gid) { sentinelClientSend(&Base.base, sizeof(fileutils_dchgrp), PtrsIn); }
	int RC;
	sentinelInPtr PtrsIn[2] = {
		{ &Str, -1 },
		nullptr
	};
};

struct fileutils_getgrnam {
	sentinelClientMessage Base;
	char *Str;
	fileutils_getgrnam(pipelineRedir redir, char *str) : Base(redir, FILEUTILS_GETGRNAM, SENTINELFLOW_WAIT, SENTINEL_CHUNK), Str(str) { sentinelClientSend(&Base.base, sizeof(fileutils_getgrnam), PtrsIn); }
	struct group *RC;
	sentinelInPtr PtrsIn[2] = {
		{ &Str, -1 },
		nullptr
	};
};

struct fileutils_dchmod {
	sentinelClientMessage Base;
	char *Str; int Mode;
	fileutils_dchmod(pipelineRedir redir, char *str, int mode) : Base(redir, FILEUTILS_DCHMOD, SENTINELFLOW_WAIT, SENTINEL_CHUNK), Str(str), Mode(mode) { sentinelClientSend(&Base.base, sizeof(fileutils_dchmod), PtrsIn); }
	int RC;
	sentinelInPtr PtrsIn[2] = {
		{ &Str, -1 },
		nullptr
	};
};

struct fileutils_dchown {
	sentinelClientMessage Base;
	char *Str; int Uid;
	fileutils_dchown(pipelineRedir redir, char *str, int uid) : Base(redir, FILEUTILS_DCHOWN, SENTINELFLOW_WAIT, SENTINEL_CHUNK), Str(str), Uid(uid) { sentinelClientSend(&Base.base, sizeof(fileutils_dchown), PtrsIn); }
	int RC;
	sentinelInPtr PtrsIn[2] = {
		{ &Str, -1 },
		nullptr
	};
};

struct fileutils_getpwnam {
	sentinelClientMessage Base;
	char *Str;
	fileutils_getpwnam(pipelineRedir redir, char *str) : Base(redir, FILEUTILS_GETPWNAM, SENTINELFLOW_WAIT, SENTINEL_CHUNK), Str(str) { sentinelClientSend(&Base.base, sizeof(fileutils_getpwnam), PtrsIn); }
	struct passwd *RC;
	sentinelInPtr PtrsIn[2] = {
		{ &Str, -1 },
		nullptr
	};
};

struct fileutils_dcmp {
	sentinelClientMessage Base;
	char *Str; char *Str2;
	fileutils_dcmp(pipelineRedir redir, char *str, char *str2) : Base(redir, FILEUTILS_DCMP, SENTINELFLOW_WAIT, SENTINEL_CHUNK), Str(str), Str2(str2) { sentinelClientSend(&Base.base, sizeof(fileutils_dcmp), PtrsIn); }
	int RC;
	sentinelInPtr PtrsIn[3] = {
		{ &Str, -1 },
		{ &Str2, -1 },
		nullptr
	};
};

struct fileutils_dcp {
	sentinelClientMessage Base;
	char *Str; char *Str2; bool SetModes;
	fileutils_dcp(pipelineRedir redir, char *str, char *str2, bool setModes) : Base(redir, FILEUTILS_DCP, SENTINELFLOW_WAIT, SENTINEL_CHUNK), Str(str), Str2(str2), SetModes(setModes) { sentinelClientSend(&Base.base, sizeof(fileutils_dcp), PtrsIn); }
	int RC;
	sentinelInPtr PtrsIn[2] = {
		{ &Str, -1 },
		nullptr
	};
};

struct fileutils_isadir {
	sentinelClientMessage Base;
	char *Str;
	fileutils_isadir(pipelineRedir redir, char *str) : Base(redir, FILEUTILS_ISADIR, SENTINELFLOW_WAIT, SENTINEL_CHUNK), Str(str) { sentinelClientSend(&Base.base, sizeof(fileutils_isadir), PtrsIn); }
	bool RC;
	sentinelInPtr PtrsIn[2] = {
		{ &Str, -1 },
		nullptr
	};
};

struct fileutils_dgrep {
	sentinelClientMessage Base;
	char *Str; char *Str2; bool IgnoreCase; bool TellName; bool TellLine;
	fileutils_dgrep(pipelineRedir redir, char *str, char *str2, bool ignoreCase, bool tellName, bool tellLine) : Base(redir, FILEUTILS_DGREP, SENTINELFLOW_WAIT, SENTINEL_CHUNK), Str(str), Str2(str2), IgnoreCase(ignoreCase), TellName(tellName), TellLine(tellLine) { sentinelClientSend(&Base.base, sizeof(fileutils_dgrep), PtrsIn); }
	int RC;
	sentinelInPtr PtrsIn[3] = {
		{ &Str, -1 },
		{ &Str2, -1 },
		nullptr
	};
};

struct fileutils_dls {
	sentinelClientMessage Base;
	char *Str; int Flags; bool EndSlash;
	fileutils_dls(pipelineRedir redir, char *str, int flags, bool endSlash) : Base(redir, FILEUTILS_DLS, SENTINELFLOW_WAIT, SENTINEL_CHUNK), Str(str), Flags(flags), EndSlash(endSlash) { sentinelClientSend(&Base.base, sizeof(fileutils_dls), PtrsIn); }
	int RC;
	sentinelInPtr PtrsIn[2] = {
		{ &Str, -1 },
		nullptr
	};
};

struct fileutils_dmkdir {
	sentinelClientMessage Base;
	char *Str; unsigned short Mode;
	fileutils_dmkdir(pipelineRedir redir, char *str, unsigned short mode) : Base(redir, FILEUTILS_DMKDIR, SENTINELFLOW_WAIT, SENTINEL_CHUNK), Str(str), Mode(mode) { sentinelClientSend(&Base.base, sizeof(fileutils_dmkdir), PtrsIn); }
	int RC;
	sentinelInPtr PtrsIn[2] = {
		{ &Str, -1 },
		nullptr
	};
};

struct fileutils_dmore {
	sentinelClientMessage Base;
	char *Str; int Fd;
	fileutils_dmore(pipelineRedir redir, char *str, int fd) : Base(redir, FILEUTILS_DMORE, SENTINELFLOW_WAIT, SENTINEL_CHUNK), Str(str), Fd(fd) { sentinelClientSend(&Base.base, sizeof(fileutils_dmore), PtrsIn); }
	int RC;
	sentinelInPtr PtrsIn[2] = {
		{ &Str, -1 },
		nullptr
	};
};

struct fileutils_dmv {
	sentinelClientMessage Base;
	char *Str; char *Str2;
	fileutils_dmv(pipelineRedir redir, char *str, char *str2) : Base(redir, FILEUTILS_DMV, SENTINELFLOW_WAIT, SENTINEL_CHUNK), Str(str), Str2(str2) { sentinelClientSend(&Base.base, sizeof(fileutils_dmv), PtrsIn); }
	int RC;
	sentinelInPtr PtrsIn[3] = {
		{ &Str, -1 },
		{ &Str2, -1 },
		nullptr
	};
};

struct fileutils_drm {
	sentinelClientMessage Base;
	char *Str;
	fileutils_drm(pipelineRedir redir, char *str) : Base(redir, FILEUTILS_DRM, SENTINELFLOW_WAIT, SENTINEL_CHUNK), Str(str) { sentinelClientSend(&Base.base, sizeof(fileutils_drm), PtrsIn); }
	int RC;
	sentinelInPtr PtrsIn[2] = {
		{ &Str, -1 },
		nullptr
	};
};

struct fileutils_drmdir {
	sentinelClientMessage Base;
	char *Str;
	fileutils_drmdir(pipelineRedir redir, char *str) : Base(redir, FILEUTILS_DRMDIR, SENTINELFLOW_WAIT, SENTINEL_CHUNK), Str(str) { sentinelClientSend(&Base.base, sizeof(fileutils_drmdir), PtrsIn); }
	int RC;
	sentinelInPtr PtrsIn[2] = {
		{ &Str, -1 },
		nullptr
	};
};

struct fileutils_dpwd {
	static __forceinline__ __device__ char *Prepare(fileutils_dpwd *t, char *data, char *dataEnd, intptr_t offset) {
		t->Ptr = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += SENTINEL_CHUNK);
		if (end > dataEnd) return nullptr;
		return end;
	}
	sentinelClientMessage Base;
	fileutils_dpwd(pipelineRedir redir) : Base(redir, FILEUTILS_DPWD, SENTINELFLOW_WAIT, SENTINEL_CHUNK, SENTINELPREPARE(Prepare)) { sentinelClientSend(&Base.base, sizeof(fileutils_dpwd)); }
	int RC;
	char *Ptr;
};

struct fileutils_dcd {
	sentinelClientMessage Base;
	char *Str;
	fileutils_dcd(pipelineRedir redir, char *str) : Base(redir, FILEUTILS_DCD, SENTINELFLOW_WAIT, SENTINEL_CHUNK), Str(str) { sentinelClientSend(&Base.base, sizeof(fileutils_dcd), PtrsIn); }
	int RC;
	sentinelInPtr PtrsIn[2] = {
		{ &Str, -1 },
		nullptr
	};
};

#endif  /* _SENTINEL_FILEUTILSMSG_H */