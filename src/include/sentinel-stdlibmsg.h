/*
sentinel-stdlibmsg.h - messages for sentinel
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
#ifndef _SENTINEL_STDLIBMSG_H
#define _SENTINEL_STDLIBMSG_H

#include <sentinel.h>
#include <stringcu.h>

enum {
	STDLIB_EXIT = 30,
	STDLIB_SYSTEM,
	STDLIB_GETENV,
	STDLIB_SETENV,
	STDLIB_UNSETENV,
	STDLIB_MKTEMP,
	STDLIB_MKSTEMP,
};

struct stdlib_exit {
	sentinelMessage Base;
	bool Std; int Status;
	__device__ stdlib_exit(bool std, int status) : Base(STDLIB_EXIT, FLOW_WAIT), Std(std), Status(status) { sentinelDeviceSend(&Base, sizeof(stdlib_exit)); }
};

struct stdlib_system {
	sentinelMessage Base;
	const char *Str;
	__device__ stdlib_system(const char *str) : Base(STDLIB_SYSTEM, FLOW_WAIT, SENTINEL_CHUNK), Str(str) { sentinelDeviceSend(&Base, sizeof(stdlib_system), PtrsIn); }
	int RC;
	sentinelInPtr PtrsIn[2] = {
		{ &Str, -1 },
		nullptr
	};
};

struct stdlib_getenv {
	static __forceinline__ __host__ char *HostPrepare(stdlib_getenv *t, char *data, char *dataEnd, intptr_t offset) {
		if (!t->RC) return data;
		int ptrLength = t->RC ? (int)strlen(t->RC) + 1 : 0;
		if (ptrLength > SENTINEL_CHUNK) { ptrLength = SENTINEL_CHUNK; t->RC[ptrLength] = 0; }
		char *ptr = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += ptrLength);
		if (end > dataEnd) return nullptr;
		memcpy(ptr, t->RC, ptrLength);
		t->RC = (char *)(ptr - offset);
		return end;
	}
	sentinelMessage Base;
	const char *Str;
	__device__ stdlib_getenv(const char *str) : Base(STDLIB_GETENV, FLOW_WAIT, SENTINEL_CHUNK), Str(str) { sentinelDeviceSend(&Base, sizeof(stdlib_getenv), PtrsIn); }
	char *RC;
	sentinelInPtr PtrsIn[2] = {
		{ &Str, -1 },
		nullptr
	};
};

struct stdlib_setenv {
	sentinelMessage Base;
	const char *Str; const char *Str2; int Replace;
	__device__ stdlib_setenv(const char *str, const char *str2, int replace) : Base(STDLIB_SETENV, FLOW_WAIT, SENTINEL_CHUNK), Str(str), Str2(str2), Replace(replace) { sentinelDeviceSend(&Base, sizeof(stdlib_setenv), PtrsIn); }
	int RC;
	sentinelInPtr PtrsIn[3] = {
		{ &Str, -1 },
		{ &Str2, -1 },
		nullptr
	};
};

struct stdlib_unsetenv {
	sentinelMessage Base;
	const char *Str;
	__device__ stdlib_unsetenv(const char *str) : Base(STDLIB_UNSETENV, FLOW_WAIT, SENTINEL_CHUNK), Str(str) { sentinelDeviceSend(&Base, sizeof(stdlib_unsetenv), PtrsIn); }
	int RC;
	sentinelInPtr PtrsIn[2] = {
		{ &Str, -1 },
		nullptr
	};
};

struct stdlib_mktemp {
	sentinelMessage Base;
	char *Str;
	__device__ stdlib_mktemp(char *str) : Base(STDLIB_MKTEMP, FLOW_WAIT, SENTINEL_CHUNK), Str(str) { sentinelDeviceSend(&Base, sizeof(stdlib_mktemp), PtrsIn); }
	char *RC;
	sentinelInPtr PtrsIn[2] = {
		{ &Str, -1 },
		nullptr
	};
};

struct stdlib_mkstemp {
	static __forceinline__ __device__ bool Postfix(stdlib_mkstemp *t, intptr_t offset) {
		char *ptr = (char *)t->Ptr - offset;
		if (t->Str) strcpy(t->Str, ptr);
		return true;
	}
	sentinelMessage Base;
	char *Str;
	__device__ stdlib_mkstemp(char *str) : Base(STDLIB_MKSTEMP, FLOW_WAIT, SENTINEL_CHUNK, nullptr, SENTINELPOSTFIX(Postfix)), Str(str) { sentinelDeviceSend(&Base, sizeof(stdlib_mkstemp), PtrsIn); }
	int RC;
	void *Ptr;
	sentinelInPtr PtrsIn[2] = {
		{ &Str, -1 },
		nullptr
	};
	//sentinelOutPtr PtrsIn[2] = {
	//	{ &Ptr, &Str, -1 },
	//	nullptr
	//};
};

#endif  /* _SENTINEL_STDLIBMSG_H */