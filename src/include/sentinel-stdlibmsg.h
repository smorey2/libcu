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
	STDLIB_MBLEN,
	STDLIB_MBTOWC,
	STDLIB_WCTOMB,
	STDLIB_MBSTOWCS,
	STDLIB_WCSTOMBS,
};

struct stdlib_exit {
	sentinelMessage Base;
	bool Std; int Status;
	__device__ stdlib_exit(bool std, int status) : Base(true, STDLIB_EXIT), Std(std), Status(status) { sentinelDeviceSend(&Base, sizeof(stdlib_exit)); }
};

struct stdlib_system {
	static __forceinline__ __device__ char *Prepare(stdlib_system *t, char *data, char *dataEnd, intptr_t offset) {
		int strLength = t->Str ? (int)strlen(t->Str) + 1 : 0;
		char *str = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += strLength);
		if (end > dataEnd) return nullptr;
		memcpy(str, t->Str, strLength);
		t->Str = str + offset;
		return end;
	}
	sentinelMessage Base;
	const char *Str;
	__device__ stdlib_system(const char *str) : Base(true, STDLIB_SYSTEM, 1024, SENTINELPREPARE(Prepare)), Str(str) { sentinelDeviceSend(&Base, sizeof(stdlib_system)); }
	int RC;
};

struct stdlib_getenv {
	static __forceinline__ __device__ char *Prepare(stdlib_getenv *t, char *data, char *dataEnd, intptr_t offset) {
		int strLength = t->Str ? (int)strlen(t->Str) + 1 : 0;
		char *str = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += strLength);
		if (end > dataEnd) return nullptr;
		memcpy(str, t->Str, strLength);
		t->Str = str + offset;
		return end;
	}
	sentinelMessage Base;
	const char *Str;
	__device__ stdlib_getenv(const char *str) : Base(true, STDLIB_GETENV, 1024, SENTINELPREPARE(Prepare)), Str(str) { sentinelDeviceSend(&Base, sizeof(stdlib_getenv)); }
	char *RC;
};

struct stdlib_setenv {
	static __forceinline__ __device__ char *Prepare(stdlib_setenv *t, char *data, char *dataEnd, intptr_t offset) {
		int strLength = t->Str ? (int)strlen(t->Str) + 1 : 0;
		int str2Length = t->Str2 ? (int)strlen(t->Str2) + 1 : 0;
		char *str = (char *)(data += ROUND8_(sizeof(*t)));
		char *str2 = (char *)(data += strLength);
		char *end = (char *)(data += str2Length);
		if (end > dataEnd) return nullptr;
		memcpy(str, t->Str, strLength);
		memcpy(str2, t->Str2, str2Length);
		t->Str = str + offset;
		t->Str2 = str2 + offset;
		return end;
	}
	sentinelMessage Base;
	const char *Str; const char *Str2; int Replace;
	__device__ stdlib_setenv(const char *str, const char *str2, int replace) : Base(true, STDLIB_SETENV, 1024, SENTINELPREPARE(Prepare)), Str(str), Str2(str2), Replace(replace) { sentinelDeviceSend(&Base, sizeof(stdlib_setenv)); }
	int RC;
};

struct stdlib_unsetenv {
	static __forceinline__ __device__ char *Prepare(stdlib_unsetenv *t, char *data, char *dataEnd, intptr_t offset) {
		int strLength = t->Str ? (int)strlen(t->Str) + 1 : 0;
		char *str = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += strLength);
		if (end > dataEnd) return nullptr;
		memcpy(str, t->Str, strLength);
		t->Str = str + offset;
		return end;
	}
	sentinelMessage Base;
	const char *Str;
	__device__ stdlib_unsetenv(const char *str) : Base(true, STDLIB_UNSETENV, 1024, SENTINELPREPARE(Prepare)), Str(str) { sentinelDeviceSend(&Base, sizeof(stdlib_unsetenv)); }
	int RC;
};

struct stdlib_mktemp {
	static __forceinline__ __device__ char *Prepare(stdlib_mktemp *t, char *data, char *dataEnd, intptr_t offset) {
		int strLength = t->Str ? (int)strlen(t->Str) + 1 : 0;
		char *str = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += strLength);
		if (end > dataEnd) return nullptr;
		memcpy(str, t->Str, strLength);
		t->Str = str + offset;
		return end;
	}
	sentinelMessage Base;
	char *Str;
	__device__ stdlib_mktemp(char *str) : Base(true, STDLIB_MKTEMP, 1024, SENTINELPREPARE(Prepare)), Str(str) { sentinelDeviceSend(&Base, sizeof(stdlib_mktemp)); }
	char *RC;
};

struct stdlib_mkstemp {
	static __forceinline__ __device__ char *Prepare(stdlib_mkstemp *t, char *data, char *dataEnd, intptr_t offset) {
		int strLength = t->Str ? (int)strlen(t->Str) + 1 : 0;
		char *str = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += strLength);
		if (end > dataEnd) return nullptr;
		memcpy(str, t->Str, strLength);
		t->Str = str + offset;
		return end;
	}
	static __forceinline__ __device__ bool Postfix(stdlib_mkstemp *t, intptr_t offset) {
		char *ptr = (char *)t->Ptr - offset;
		strcpy(t->Str, ptr);
		return true;
	}
	sentinelMessage Base;
	char *Str;
	__device__ stdlib_mkstemp(char *str) : Base(true, STDLIB_MKSTEMP, 1024, SENTINELPREPARE(Prepare), SENTINELPOSTFIX(Postfix)), Str(str) { sentinelDeviceSend(&Base, sizeof(stdlib_mkstemp)); }
	int RC;
	void *Ptr;
};

//struct stdlib_mblen {
//	static __forceinline__ __device__ char *Prepare(stdlib_mblen *t, char *data, char *dataEnd, intptr_t offset) {
//		int strLength = t->Str ? (int)strlen(t->Str) + 1 : 0;
//		char *str = (char *)(data += ROUND8_(sizeof(*t)));
//		char *end = (char *)(data += strLength);
//		if (end > dataEnd) return nullptr;
//		memcpy(str, t->Str, strLength);
//		t->Str = str + offset;
//		return end;
//	}
//	sentinelMessage Base;
//	const char *Str; size_t Size;
//	__device__ stdlib_mblen(const char *str, size_t size) : Base(true, STDLIB_MBLEN, 1024, SENTINELPREPARE(Prepare)), Str(str), Size(size) { sentinelDeviceSend(&Base, sizeof(stdlib_mblen)); }
//	int RC;
//};
//
//struct stdlib_mbtowc {
//	static __forceinline__ __device__ char *Prepare(stdlib_mbtowc *t, char *data, char *dataEnd, intptr_t offset) {
//		return 0;
//		//int strLength = t->Str ? (int)strlen(t->Str) + 1 : 0;
//		//char *str = (char *)(data += ROUND8_(sizeof(*t)));
//		//char *end = (char *)(data += strLength);
//		//if (end > dataEnd) return nullptr;
//		//memcpy(str, t->Str, strLength);
//		//t->Str = str + offset;
//		//return end;
//	}
//	//static __forceinline__ __device__ char *Prepare(stdio_fread *t, char *data, char *dataEnd, intptr_t offset) {
//	//	char *ptr = (char *)(data += ROUND8_(sizeof(*t)));
//	//	char *end = (char *)(data += 1024);
//	//	if (end > dataEnd) return nullptr;
//	//	t->Ptr = ptr + offset;
//	//	return end;
//	//}
//	static __forceinline__ __device__ bool Postfix(stdlib_mbtowc *t, intptr_t offset) {
//		char *ptr = (char *)t->Ptr - offset;
//		if ((int)t->RC > 0) memcpy(t->Buf, ptr, t->RC);
//		return true;
//	}
//	sentinelMessage Base;
//	wchar_t *Str; void *Buf; size_t Size;
//	__device__ stdlib_mbtowc(wchar_t *str, void *buf, size_t size) : Base(true, STDLIB_MBTOWC, 1024, SENTINELPREPARE(Prepare), SENTINELPOSTFIX(Postfix)), Str(str), Buf(buf), Size(size) { sentinelDeviceSend(&Base, sizeof(stdlib_mbtowc)); }
//	int RC;
//	void *Ptr;
//};
//
//struct stdlib_wctomb {
//	static __forceinline__ __device__ char *Prepare(stdlib_wctomb *t, char *data, char *dataEnd, intptr_t offset) {
//		int strLength = t->Str ? (int)strlen(t->Str) + 1 : 0;
//		char *str = (char *)(data += ROUND8_(sizeof(*t)));
//		char *end = (char *)(data += strLength);
//		if (end > dataEnd) return nullptr;
//		memcpy(str, t->Str, strLength);
//		t->Str = str + offset;
//		return end;
//	}
//	sentinelMessage Base;
//	char *Str;
//	__device__ stdlib_wctomb(char *str) : Base(true, STDLIB_WCTOMB, 1024, SENTINELPREPARE(Prepare)), Str(str) { sentinelDeviceSend(&Base, sizeof(stdlib_wctomb)); }
//	int RC;
//};
//
//struct stdlib_mbstowcs {
//	static __forceinline__ __device__ char *Prepare(stdlib_mbstowcs *t, char *data, char *dataEnd, intptr_t offset) {
//		int strLength = t->Str ? (int)strlen(t->Str) + 1 : 0;
//		char *str = (char *)(data += ROUND8_(sizeof(*t)));
//		char *end = (char *)(data += strLength);
//		if (end > dataEnd) return nullptr;
//		memcpy(str, t->Str, strLength);
//		t->Str = str + offset;
//		return end;
//	}
//	sentinelMessage Base;
//	char *Str;
//	__device__ stdlib_mbstowcs(char *str) : Base(true, STDLIB_MBSTOWCS, 1024, SENTINELPREPARE(Prepare)), Str(str) { sentinelDeviceSend(&Base, sizeof(stdlib_mbstowcs)); }
//	int RC;
//};
//
//
//struct stdlib_wcstombs {
//	static __forceinline__ __device__ char *Prepare(stdlib_wcstombs *t, char *data, char *dataEnd, intptr_t offset) {
//		int strLength = t->Str ? (int)strlen(t->Str) + 1 : 0;
//		char *str = (char *)(data += ROUND8_(sizeof(*t)));
//		char *end = (char *)(data += strLength);
//		if (end > dataEnd) return nullptr;
//		memcpy(str, t->Str, strLength);
//		t->Str = str + offset;
//		return end;
//	}
//	sentinelMessage Base;
//	char *Str;
//	__device__ stdlib_wcstombs(char *str) : Base(true, STDLIB_WCSTOMBS, 1024, SENTINELPREPARE(Prepare)), Str(str) { sentinelDeviceSend(&Base, sizeof(stdlib_wcstombs)); }
//	int RC;
//};

#endif  /* _SENTINEL_STDLIBMSG_H */