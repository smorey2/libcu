/*
sentinel-unistdmsg.h - messages for sentinel
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
#ifndef _SENTINEL_UNISTDMSG_H
#define _SENTINEL_UNISTDMSG_H

#include <sentinel.h>
#include <crtdefscu.h>
#include <stringcu.h>

enum {
	UNISTD_ACCESS = 42,
	UNISTD_LSEEK,
	UNISTD_CLOSE,
	UNISTD_READ,
	UNISTD_WRITE,
	UNISTD_CHOWN,
	UNISTD_CHDIR,
	UNISTD_GETCWD,
	UNISTD_DUP,
	UNISTD_UNLINK,
	UNISTD_RMDIR,
};

struct unistd_access {
	sentinelMessage Base;
	const char *Str; int Type;
	__device__ unistd_access(const char *str, int type) : Base(UNISTD_ACCESS, SENTINELFLOW_WAIT, SENTINEL_CHUNK), Str(str), Type(type) { sentinelDeviceSend(&Base, sizeof(unistd_access), PtrsIn); }
	int RC;
	sentinelInPtr PtrsIn[2] = {
		{ &Str, -1 },
		nullptr
	};
};

struct unistd_lseek {
	sentinelMessage Base;
	int Handle; long long Offset; int Whence; bool Bit64;
	__device__ unistd_lseek(int fd, long long offset, int whence, bool bit64) : Base(UNISTD_LSEEK, SENTINELFLOW_WAIT), Handle(fd), Offset(offset), Whence(whence), Bit64(bit64) { sentinelDeviceSend(&Base, sizeof(unistd_lseek)); }
	long long RC;
};

struct unistd_close {
	sentinelMessage Base;
	int Handle;
	__device__ unistd_close(int fd) : Base(UNISTD_CLOSE, SENTINELFLOW_WAIT), Handle(fd) { sentinelDeviceSend(&Base, sizeof(unistd_close)); }
	int RC;
};

struct unistd_read {
	sentinelMessage Base;
	int Handle; void *Buf; size_t Size;
	__device__ unistd_read(bool wait, int fd, void *buf, size_t size) : Base(UNISTD_READ, wait ? SENTINELFLOW_WAIT : SENTINELFLOW_NONE, SENTINEL_CHUNK), Handle(fd), Buf(buf), Size(size) { PtrsOut[0].size = size; sentinelDeviceSend(&Base, sizeof(unistd_read), nullptr, PtrsOut); }
	size_t RC;
	void *Ptr;
	sentinelOutPtr PtrsOut[2] = {
		{ &Ptr, &Buf, 0 },
		nullptr
	};
};

struct unistd_write {
	sentinelMessage Base;
	int Handle; const void *Ptr; size_t Size;
	__device__ unistd_write(bool wait, int fd, const void *ptr, size_t size) : Base(UNISTD_WRITE, wait ? SENTINELFLOW_WAIT : SENTINELFLOW_NONE, SENTINEL_CHUNK), Handle(fd), Ptr(ptr), Size(size) { PtrsIn[0].size = size; sentinelDeviceSend(&Base, sizeof(unistd_write), PtrsIn); }
	size_t RC;
	sentinelInPtr PtrsIn[2] = {
		{ &Ptr, 0 },
		nullptr
	};
};

struct unistd_chown {
	sentinelMessage Base;
	const char *Str; int Owner; int Group;
	__device__ unistd_chown(const char *str, int owner, int group) : Base(UNISTD_CHOWN, SENTINELFLOW_WAIT, SENTINEL_CHUNK), Str(str), Owner(owner), Group(group) { sentinelDeviceSend(&Base, sizeof(unistd_chown), PtrsIn); }
	int RC;
	sentinelInPtr PtrsIn[2] = {
		{ &Str, -1 },
		nullptr
	};
};

struct unistd_chdir {
	sentinelMessage Base;
	const char *Str;
	__device__ unistd_chdir(const char *str) : Base(UNISTD_CHDIR, SENTINELFLOW_WAIT, SENTINEL_CHUNK), Str(str) { sentinelDeviceSend(&Base, sizeof(unistd_chdir), PtrsIn); }
	int RC;
	sentinelInPtr PtrsIn[2] = {
		{ &Str, -1 },
		nullptr
	};
};

struct unistd_getcwd {
	sentinelMessage Base;
	char *Ptr; size_t Size;
	__device__ unistd_getcwd(char *buf, size_t size) : Base(UNISTD_GETCWD, SENTINELFLOW_WAIT, SENTINEL_CHUNK), Ptr(buf), Size(size) { sentinelDeviceSend(&Base, sizeof(unistd_getcwd), nullptr, PtrsOut); }
	char *RC;
	sentinelOutPtr PtrsOut[2] = {
		{ &Ptr, nullptr, -1 },
		nullptr
	};
};

struct unistd_dup {
	sentinelMessage Base;
	int Handle; int Handle2; bool Dup1;
	__device__ unistd_dup(int fd, int fd2, bool dup1) : Base(UNISTD_DUP, SENTINELFLOW_WAIT), Handle(fd), Handle2(fd2), Dup1(dup1) { sentinelDeviceSend(&Base, sizeof(unistd_dup)); }
	int RC;
};

struct unistd_unlink {
	sentinelMessage Base;
	const char *Str;
	__device__ unistd_unlink(const char *str) : Base(UNISTD_UNLINK, SENTINELFLOW_WAIT, SENTINEL_CHUNK), Str(str) { sentinelDeviceSend(&Base, sizeof(unistd_unlink), PtrsIn); }
	int RC;
	sentinelInPtr PtrsIn[2] = {
		{ &Str, -1 },
		nullptr
	};
};

struct unistd_rmdir {
	sentinelMessage Base;
	const char *Str;
	__device__ unistd_rmdir(const char *str) : Base(UNISTD_RMDIR, SENTINELFLOW_WAIT, SENTINEL_CHUNK), Str(str) { sentinelDeviceSend(&Base, sizeof(unistd_rmdir), PtrsIn); }
	int RC;
	sentinelInPtr PtrsIn[2] = {
		{ &Str, -1 },
		nullptr
	};
};

#endif  /* _SENTINEL_UNISTDMSG_H */