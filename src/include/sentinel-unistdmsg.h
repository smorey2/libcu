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
	static __forceinline__ __device__ char *Prepare(unistd_access *t, char *data, char *dataEnd, intptr_t offset) {
		int strLength = t->Str ? (int)strlen(t->Str) + 1 : 0;
		char *str = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += strLength);
		if (end > dataEnd) return nullptr;
		memcpy(str, t->Str, strLength);
		if (t->Str) t->Str = str + offset;
		return end;
	}
	sentinelMessage Base;
	const char *Str; int Type;
	__device__ unistd_access(const char *str, int type) : Base(UNISTD_ACCESS, FLOW_WAIT, SENTINEL_CHUNK, SENTINELPREPARE(Prepare)), Str(str), Type(type) { sentinelDeviceSend(&Base, sizeof(unistd_access)); }
	int RC;
};

struct unistd_lseek {
	sentinelMessage Base;
	int Handle; long long Offset; int Whence; bool Bit64;
	__device__ unistd_lseek(int fd, long long offset, int whence, bool bit64) : Base(UNISTD_LSEEK, FLOW_WAIT), Handle(fd), Offset(offset), Whence(whence), Bit64(bit64) { sentinelDeviceSend(&Base, sizeof(unistd_lseek)); }
	long long RC;
};

struct unistd_close {
	sentinelMessage Base;
	int Handle;
	__device__ unistd_close(int fd) : Base(UNISTD_CLOSE, FLOW_WAIT), Handle(fd) { sentinelDeviceSend(&Base, sizeof(unistd_close)); }
	int RC;
};

struct unistd_read {
	static __forceinline__ __device__ char *Prepare(unistd_read *t, char *data, char *dataEnd, intptr_t offset) {
		char *ptr = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += SENTINEL_CHUNK);
		if (end > dataEnd) return nullptr;
		t->Ptr = ptr + offset;
		return end;
	}
	static __forceinline__ __device__ bool Postfix(unistd_read *t, intptr_t offset) {
		char *ptr = (char *)t->Ptr - offset;
		if (t->RC > 0) memcpy(t->Buf, ptr, t->RC);
		return true;
	}
	sentinelJumboMessage Base;
	int Handle; void *Buf; size_t Size;
	__device__ unistd_read(bool wait, int fd, void *buf, size_t size) : Base(UNISTD_READ, (wait ? FLOW_WAIT : FLOW_NONE) | FLOW_JUMBOIN, SENTINEL_CHUNK, SENTINELPREPARE(Prepare), SENTINELPOSTFIX(Postfix)), Handle(fd), Buf(buf), Size(size) { sentinelDeviceSend(&Base.Base, sizeof(unistd_read)); }
	size_t RC;
	void *Ptr;
};

struct unistd_write {
	static __forceinline__ __device__ char *Prepare(unistd_write *t, char *data, char *dataEnd, intptr_t offset) {
		size_t size = t->Size;
		char *ptr = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += size);
		if (end > dataEnd) return nullptr;
		memcpy(ptr, t->Ptr, size);
		t->Ptr = (char *)ptr + offset;
		return end;
	}
	sentinelJumboMessage Base;
	int Handle; const void *Ptr; size_t Size;
	__device__ unistd_write(bool wait, int fd, const void *ptr, size_t size) : Base(UNISTD_WRITE, (wait ? FLOW_WAIT : FLOW_NONE) | FLOW_JUMBOOUT, SENTINEL_CHUNK, SENTINELPREPARE(Prepare)), Handle(fd), Ptr(ptr), Size(size) { sentinelDeviceSend(&Base.Base, sizeof(unistd_write)); }
	size_t RC;
};

struct unistd_chown {
	static __forceinline__ __device__ char *Prepare(unistd_chown *t, char *data, char *dataEnd, intptr_t offset) {
		int strLength = t->Str ? (int)strlen(t->Str) + 1 : 0;
		char *str = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += strLength);
		if (end > dataEnd) return nullptr;
		memcpy(str, t->Str, strLength);
		if (t->Str) t->Str = str + offset;
		return end;
	}
	sentinelMessage Base;
	const char *Str; int Owner; int Group;
	__device__ unistd_chown(const char *str, int owner, int group) : Base(UNISTD_CHOWN, FLOW_WAIT, SENTINEL_CHUNK, SENTINELPREPARE(Prepare)), Str(str), Owner(owner), Group(group) { sentinelDeviceSend(&Base, sizeof(unistd_chown)); }
	int RC;
};

struct unistd_chdir {
	static __forceinline__ __device__ char *Prepare(unistd_chdir *t, char *data, char *dataEnd, intptr_t offset) {
		int strLength = t->Str ? (int)strlen(t->Str) + 1 : 0;
		char *str = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += strLength);
		if (end > dataEnd) return nullptr;
		memcpy(str, t->Str, strLength);
		if (t->Str) t->Str = str + offset;
		return end;
	}
	sentinelMessage Base;
	const char *Str;
	__device__ unistd_chdir(const char *str) : Base(UNISTD_CHDIR, FLOW_WAIT, SENTINEL_CHUNK, SENTINELPREPARE(Prepare)), Str(str) { sentinelDeviceSend(&Base, sizeof(unistd_chdir)); }
	int RC;
};

struct unistd_getcwd {
	static __forceinline__ __device__ char *Prepare(unistd_getcwd *t, char *data, char *dataEnd, intptr_t offset) {
		t->Ptr = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += SENTINEL_CHUNK);
		if (end > dataEnd) return nullptr;
		return end;
	}
	sentinelMessage Base;
	char *Ptr; size_t Size;
	__device__ unistd_getcwd(char *buf, size_t size) : Base(UNISTD_GETCWD, FLOW_WAIT, SENTINEL_CHUNK, SENTINELPREPARE(Prepare)), Ptr(buf), Size(size) { sentinelDeviceSend(&Base, sizeof(unistd_getcwd)); }
	char *RC;
};

struct unistd_dup {
	sentinelMessage Base;
	int Handle; int Handle2; bool Dup1;
	__device__ unistd_dup(int fd, int fd2, bool dup1) : Base(UNISTD_DUP, FLOW_WAIT), Handle(fd), Handle2(fd2), Dup1(dup1) { sentinelDeviceSend(&Base, sizeof(unistd_dup)); }
	int RC;
};

struct unistd_unlink {
	static __forceinline__ __device__ char *Prepare(unistd_unlink *t, char *data, char *dataEnd, intptr_t offset) {
		int strLength = t->Str ? (int)strlen(t->Str) + 1 : 0;
		char *str = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += strLength);
		if (end > dataEnd) return nullptr;
		memcpy(str, t->Str, strLength);
		if (t->Str) t->Str = str + offset;
		return end;
	}
	sentinelMessage Base;
	const char *Str;
	__device__ unistd_unlink(const char *str) : Base(UNISTD_UNLINK, FLOW_WAIT, SENTINEL_CHUNK, SENTINELPREPARE(Prepare)), Str(str) { sentinelDeviceSend(&Base, sizeof(unistd_unlink)); }
	int RC;
};

struct unistd_rmdir {
	static __forceinline__ __device__ char *Prepare(unistd_rmdir *t, char *data, char *dataEnd, intptr_t offset) {
		int strLength = t->Str ? (int)strlen(t->Str) + 1 : 0;
		char *str = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += strLength);
		if (end > dataEnd) return nullptr;
		memcpy(str, t->Str, strLength);
		if (t->Str) t->Str = str + offset;
		return end;
	}
	sentinelMessage Base;
	const char *Str;
	__device__ unistd_rmdir(const char *str) : Base(UNISTD_RMDIR, FLOW_WAIT, SENTINEL_CHUNK, SENTINELPREPARE(Prepare)), Str(str) { sentinelDeviceSend(&Base, sizeof(unistd_rmdir)); }
	int RC;
};

#endif  /* _SENTINEL_UNISTDMSG_H */