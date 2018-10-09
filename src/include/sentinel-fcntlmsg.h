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
	__device__ fcntl_fcntl(int fd, int cmd, int p0, bool bit64)
		: Base(true, FCNTL_FCNTL), Handle(fd), Cmd(cmd), P0(p0), Bit64(bit64) {
		sentinelDeviceSend(&Base, sizeof(fcntl_fcntl));
	}
	int RC;
};

struct fcntl_open {
	static __forceinline__ __device__ char *Prepare(fcntl_open *t, char *data, char *dataEnd, intptr_t offset) {
		int strLength = t->Str ? (int)strlen(t->Str) + 1 : 0;
		char *str = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += strLength);
		if (end > dataEnd) return nullptr;
		memcpy(str, t->Str, strLength);
		if (t->Str) t->Str = str + offset;
		return end;
	}
	sentinelMessage Base;
	const char *Str; int OFlag; int P0; bool Bit64;
	__device__ fcntl_open(const char *str, int oflag, int p0, bool bit64) : Base(true, FCNTL_OPEN, SENTINEL_CHUNK, SENTINELPREPARE(Prepare)), Str(str), OFlag(oflag), P0(p0), Bit64(bit64) { sentinelDeviceSend(&Base, sizeof(fcntl_open)); }
	int RC;
};

struct fcntl_stat {
	static __forceinline__ __device__ char *Prepare(fcntl_stat *t, char *data, char *dataEnd, intptr_t offset) {
		int strLength = t->Str ? (int)strlen(t->Str) + 1 : 0;
		char *str = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += strLength);
		if (end > dataEnd) return nullptr;
		memcpy(str, t->Str, strLength);
		if (t->Str) t->Str = str + offset;
		t->Ptr = str + offset;
		return end;
	}
	static __forceinline__ __device__ bool Postfix(fcntl_stat *t, intptr_t offset) {
		char *ptr = (char *)t->Ptr - offset;
		if (!t->Bit64) memcpy(t->Buf, ptr, sizeof(struct stat));
		else memcpy(t->Buf64, ptr, sizeof(struct _stat64));
		return true;
	}
	sentinelMessage Base;
	const char *Str; struct stat *Buf; struct _stat64 *Buf64; bool Bit64; bool LStat;
	__device__ fcntl_stat(const char *str, struct stat *buf, struct _stat64 *buf64, bool bit64, bool lstat_) : Base(true, FCNTL_STAT, SENTINEL_CHUNK, SENTINELPREPARE(Prepare), SENTINELPOSTFIX(Postfix)), Str(str), Buf(buf), Buf64(buf64), Bit64(bit64), LStat(lstat_) { sentinelDeviceSend(&Base, sizeof(fcntl_stat)); }
	int RC;
	void *Ptr;
};

struct fcntl_fstat {
	static __forceinline__ __device__ char *Prepare(fcntl_fstat *t, char *data, char *dataEnd, intptr_t offset) {
		char *ptr = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += SENTINEL_CHUNK);
		if (end > dataEnd) return nullptr;
		t->Ptr = ptr + offset;
		return end;
	}
	static __forceinline__ __device__ bool Postfix(fcntl_fstat *t, intptr_t offset) {
		char *ptr = (char *)t->Ptr - offset;
		if (!t->Bit64) memcpy(t->Buf, ptr, sizeof(struct stat));
		else memcpy(t->Buf64, ptr, sizeof(struct _stat64));
		return true;
	}
	sentinelMessage Base;
	int Handle; struct stat *Buf; struct _stat64 *Buf64; bool Bit64;
	__device__ fcntl_fstat(int fd, struct stat *buf, struct _stat64 *buf64, bool bit64) : Base(true, FCNTL_FSTAT, SENTINEL_CHUNK, SENTINELPREPARE(Prepare), SENTINELPOSTFIX(Postfix)), Handle(fd), Buf(buf), Buf64(buf64), Bit64(bit64) { sentinelDeviceSend(&Base, sizeof(fcntl_fstat)); }
	int RC;
	void *Ptr;
};

struct fcntl_chmod {
	static __forceinline__ __device__ char *Prepare(fcntl_chmod *t, char *data, char *dataEnd, intptr_t offset) {
		int strLength = t->Str ? (int)strlen(t->Str) + 1 : 0;
		char *str = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += strLength);
		if (end > dataEnd) return nullptr;
		memcpy(str, t->Str, strLength);
		if (t->Str) t->Str = str + offset;
		return end;
	}
	sentinelMessage Base;
	const char *Str; mode_t Mode;
	__device__ fcntl_chmod(const char *str, mode_t mode) : Base(true, FCNTL_CHMOD, SENTINEL_CHUNK, SENTINELPREPARE(Prepare)), Str(str), Mode(mode) { sentinelDeviceSend(&Base, sizeof(fcntl_chmod)); }
	int RC;
};

struct fcntl_mkdir {
	static __forceinline__ __device__ char *Prepare(fcntl_mkdir *t, char *data, char *dataEnd, intptr_t offset) {
		int strLength = t->Str ? (int)strlen(t->Str) + 1 : 0;
		char *str = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += strLength);
		if (end > dataEnd) return nullptr;
		memcpy(str, t->Str, strLength);
		if (t->Str) t->Str = str + offset;
		return end;
	}
	sentinelMessage Base;
	const char *Str; mode_t Mode;
	__device__ fcntl_mkdir(const char *str, mode_t mode) : Base(true, FCNTL_MKDIR, SENTINEL_CHUNK, SENTINELPREPARE(Prepare)), Str(str), Mode(mode) { sentinelDeviceSend(&Base, sizeof(fcntl_mkdir)); }
	int RC;
};

struct fcntl_mkfifo {
	static __forceinline__ __device__ char *Prepare(fcntl_mkfifo *t, char *data, char *dataEnd, intptr_t offset) {
		int strLength = t->Str ? (int)strlen(t->Str) + 1 : 0;
		char *str = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += strLength);
		if (end > dataEnd) return nullptr;
		memcpy(str, t->Str, strLength);
		if (t->Str) t->Str = str + offset;
		return end;
	}
	sentinelMessage Base;
	const char *Str; mode_t Mode;
	__device__ fcntl_mkfifo(const char *str, mode_t mode) : Base(true, FCNTL_MKFIFO, SENTINEL_CHUNK, SENTINELPREPARE(Prepare)), Str(str), Mode(mode) { sentinelDeviceSend(&Base, sizeof(fcntl_mkfifo)); }
	int RC;
};

#endif  /* _SENTINEL_STATMSG_H */