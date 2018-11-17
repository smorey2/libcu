/*
sentinel-fcntlmsg.h - messages for sentinel
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
#ifndef _SENTINEL_FCNTLMSG_H
#define _SENTINEL_FCNTLMSG_H

#include <sentinel.h>
#include <crtdefscu.h>
#include <stringcu.h>
#if __OS_UNIX
#define _stat64 stat64
#endif

enum {
	FCNTL_FCNTL = 53,
	FCNTL_OPEN,
	FCNTL_CLOSE,
	FCNTL_STAT,
	FCNTL_FSTAT,
	FCNTL_CHMOD,
	FCNTL_MKDIR,
	FCNTL_MKFIFO,
};

struct fcntl_fcntl {
	sentinelMessage Base;
	int Handle; int Cmd; int P0; bool Bit64;
	__device__ fcntl_fcntl(int fd, int cmd, int p0, bool bit64) : Base(FCNTL_FCNTL, FLOW_WAIT), Handle(fd), Cmd(cmd), P0(p0), Bit64(bit64) { sentinelDeviceSend(&Base, sizeof(fcntl_fcntl)); }
	int RC;
};

struct fcntl_open {
	sentinelMessage Base;
	const char *Str; int OFlag; int P0; bool Bit64;
	__device__ fcntl_open(const char *str, int oflag, int p0, bool bit64) : Base(FCNTL_OPEN, FLOW_WAIT, SENTINEL_CHUNK), Str(str), OFlag(oflag), P0(p0), Bit64(bit64) { sentinelDeviceSend(&Base, sizeof(fcntl_open), PtrsIn); }
	int RC;
	sentinelInPtr PtrsIn[2] = {
		{ &Str, -1 },
		nullptr
	};
};

struct fcntl_stat {
	sentinelMessage Base;
	const char *Str; struct stat *Buf; struct _stat64 *Buf64; bool Bit64; bool LStat;
	__device__ fcntl_stat(const char *str, struct stat *buf, struct _stat64 *buf64, bool bit64, bool lstat_) : Base(FCNTL_STAT, FLOW_WAIT, SENTINEL_CHUNK), Str(str), Buf(buf), Buf64(buf64), Bit64(bit64), LStat(lstat_) { if (bit64) { PtrsOut[0].buf = &Buf64; PtrsOut[0].size = sizeof(struct _stat64); } sentinelDeviceSend(&Base, sizeof(fcntl_stat), PtrsIn, PtrsOut); }
	int RC;
	void *Ptr;
	sentinelInPtr PtrsIn[2] = {
		{ &Str, -1 },
		nullptr
	};
	sentinelOutPtr PtrsOut[2] = {
		{ &Ptr, &Buf, sizeof(struct stat) },
		nullptr
	};
};

struct fcntl_fstat {
	sentinelMessage Base;
	int Handle; struct stat *Buf; struct _stat64 *Buf64; bool Bit64;
	__device__ fcntl_fstat(int fd, struct stat *buf, struct _stat64 *buf64, bool bit64) : Base(FCNTL_FSTAT, FLOW_WAIT, SENTINEL_CHUNK), Handle(fd), Buf(buf), Buf64(buf64), Bit64(bit64) { if (bit64) { PtrsOut[0].buf = &Buf64; PtrsOut[0].size = sizeof(struct _stat64); } sentinelDeviceSend(&Base, sizeof(fcntl_fstat), nullptr, PtrsOut); }
	int RC;
	void *Ptr;
	sentinelOutPtr PtrsOut[2] = {
		{ &Ptr, &Buf, sizeof(struct stat) },
		nullptr
	};
};

struct fcntl_chmod {
	sentinelMessage Base;
	const char *Str; mode_t Mode;
	__device__ fcntl_chmod(const char *str, mode_t mode) : Base(FCNTL_CHMOD, FLOW_WAIT, SENTINEL_CHUNK), Str(str), Mode(mode) { sentinelDeviceSend(&Base, sizeof(fcntl_chmod), PtrsIn); }
	int RC;
	sentinelInPtr PtrsIn[2] = {
		{ &Str, -1 },
		nullptr
	};
};

struct fcntl_mkdir {
	sentinelMessage Base;
	const char *Str; mode_t Mode;
	__device__ fcntl_mkdir(const char *str, mode_t mode) : Base(FCNTL_MKDIR, FLOW_WAIT, SENTINEL_CHUNK), Str(str), Mode(mode) { sentinelDeviceSend(&Base, sizeof(fcntl_mkdir), PtrsIn); }
	int RC;
	sentinelInPtr PtrsIn[2] = {
		{ &Str, -1 },
		nullptr
	};
};

struct fcntl_mkfifo {
	sentinelMessage Base;
	const char *Str; mode_t Mode;
	__device__ fcntl_mkfifo(const char *str, mode_t mode) : Base(FCNTL_MKFIFO, FLOW_WAIT, SENTINEL_CHUNK), Str(str), Mode(mode) { sentinelDeviceSend(&Base, sizeof(fcntl_mkfifo), PtrsIn); }
	int RC;
	sentinelInPtr PtrsIn[2] = {
		{ &Str, -1 },
		nullptr
	};
};

#endif  /* _SENTINEL_STATMSG_H */