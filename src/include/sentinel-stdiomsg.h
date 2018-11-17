/*
sentinel-stdiomsg.h - messages for sentinel
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
#ifndef _SENTINEL_STDIOMSG_H
#define _SENTINEL_STDIOMSG_H

#include <sentinel.h>
#include <crtdefscu.h>
#include <stringcu.h>
#include <stdiocu.h>

enum {
	STDIO_REMOVE = 1,
	STDIO_RENAME,
	STDIO_FCLOSE,
	STDIO_FFLUSH,
	STDIO_FREOPEN,
	STDIO_SETVBUF,
	STDIO_FGETC,
	STDIO_FPUTC,
	STDIO_FGETS,
	STDIO_FPUTS,
	STDIO_UNGETC,
	STDIO_FREAD,
	STDIO_FWRITE,
	STDIO_FSEEK,
	STDIO_FTELL,
	STDIO_REWIND,
	STDIO_FSEEKO,
	STDIO_FTELLO,
	STDIO_FGETPOS,
	STDIO_FSETPOS,
	STDIO_CLEARERR,
	STDIO_FEOF,
	STDIO_FERROR,
	STDIO_FILENO,
};

struct stdio_remove {
	static __forceinline__ __device__ char *Prepare(stdio_remove *t, char *data, char *dataEnd, intptr_t offset) {
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
	__device__ stdio_remove(const char *str) : Base(STDIO_REMOVE, FLOW_WAIT, SENTINEL_CHUNK, SENTINELPREPARE(Prepare)), Str(str) { sentinelDeviceSend(&Base, sizeof(stdio_remove)); }
	int RC;
};

struct stdio_rename {
	static __forceinline__ __device__ char *Prepare(stdio_rename *t, char *data, char *dataEnd, intptr_t offset) {
		int strLength = t->Str ? (int)strlen(t->Str) + 1 : 0;
		int str2Length = t->Str2 ? (int)strlen(t->Str2) + 1 : 0;
		char *str = (char *)(data += ROUND8_(sizeof(*t)));
		char *str2 = (char *)(data += strLength);
		char *end = (char *)(data += str2Length);
		if (end > dataEnd) return nullptr;
		memcpy(str, t->Str, strLength);
		memcpy(str2, t->Str2, str2Length);
		if (t->Str) t->Str = str + offset;
		if (t->Str2) t->Str2 = str2 + offset;
		return end;
	}
	sentinelMessage Base;
	const char *Str; const char *Str2;
	__device__ stdio_rename(const char *str, const char *str2) : Base(STDIO_RENAME, FLOW_WAIT, SENTINEL_CHUNK, SENTINELPREPARE(Prepare)), Str(str), Str2(str2) { sentinelDeviceSend(&Base, sizeof(stdio_rename)); }
	int RC;
};

struct stdio_fclose {
	sentinelMessage Base;
	FILE *File;
	__device__ stdio_fclose(bool wait, FILE *file) : Base(STDIO_FCLOSE, wait ? FLOW_WAIT : FLOW_NONE), File(file) { sentinelDeviceSend(&Base, sizeof(stdio_fclose)); }
	int RC;
};

struct stdio_fflush {
	sentinelMessage Base;
	FILE *File;
	__device__ stdio_fflush(bool wait, FILE *file) : Base(STDIO_FFLUSH, wait ? FLOW_WAIT : FLOW_NONE), File(file) { sentinelDeviceSend(&Base, sizeof(stdio_fflush)); }
	int RC;
};

struct stdio_freopen {
	static __forceinline__ __device__ char *Prepare(stdio_freopen *t, char *data, char *dataEnd, intptr_t offset) {
		int strLength = t->Str ? (int)strlen(t->Str) + 1 : 0;
		int str2Length = t->Str2 ? (int)strlen(t->Str2) + 1 : 0;
		char *str = (char *)(data += ROUND8_(sizeof(*t)));
		char *str2 = (char *)(data += strLength);
		char *end = (char *)(data += str2Length);
		if (end > dataEnd) return nullptr;
		memcpy(str, t->Str, strLength);
		memcpy(str2, t->Str2, str2Length);
		if (t->Str) t->Str = str + offset;
		if (t->Str2) t->Str2 = str2 + offset;
		return end;
	}
	sentinelMessage Base;
	const char *Str; const char *Str2; FILE *Stream;
	__device__ stdio_freopen(const char *str, const char *str2, FILE *stream) : Base(STDIO_FREOPEN, FLOW_WAIT, SENTINEL_CHUNK, SENTINELPREPARE(Prepare)), Str(str), Str2(str2), Stream(stream) { sentinelDeviceSend(&Base, sizeof(stdio_freopen)); }
	FILE *RC;
};

struct stdio_setvbuf {
	static __forceinline__ __device__ char *Prepare(stdio_setvbuf *t, char *data, char *dataEnd, intptr_t offset) {
		int bufferLength = t->Buffer ? (int)strlen(t->Buffer) + 1 : 0;
		char *buffer = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += bufferLength);
		if (end > dataEnd) return nullptr;
		memcpy(buffer, t->Buffer, bufferLength);
		t->Buffer = buffer + offset;
		return end;
	}
	sentinelMessage Base;
	FILE *File; char *Buffer; int Mode; size_t Size;
	__device__ stdio_setvbuf(FILE *file, char *buffer, int mode, size_t size) : Base(STDIO_SETVBUF, FLOW_WAIT, SENTINEL_CHUNK, SENTINELPREPARE(Prepare)), File(file), Buffer(buffer), Mode(mode), Size(size) { sentinelDeviceSend(&Base, sizeof(stdio_setvbuf)); }
	int RC;
};

struct stdio_fgetc {
	sentinelMessage Base;
	FILE *File;
	__device__ stdio_fgetc(FILE *file) : Base(true, STDIO_FGETC), File(file) { sentinelDeviceSend(&Base, sizeof(stdio_fgetc)); }
	int RC;
};

struct stdio_fputc {
	sentinelMessage Base;
	int Ch; FILE *File;
	__device__ stdio_fputc(bool wait, int ch, FILE *file) : Base(wait, STDIO_FPUTC), Ch(ch), File(file) { sentinelDeviceSend(&Base, sizeof(stdio_fputc)); }
	int RC;
};

struct stdio_fgets {
	static __forceinline__ __device__ char *Prepare(stdio_fgets *t, char *data, char *dataEnd, intptr_t offset) {
		t->Str = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += SENTINEL_CHUNK);
		if (end > dataEnd) return nullptr;
		return end;
	}
	sentinelJumboMessage Base;
	int Num; FILE *File;
	__device__ stdio_fgets(char *str, int num, FILE *file) : Base(STDIO_FGETS, FLOW_WAIT | FLOW_JUMBOIN, SENTINEL_CHUNK, SENTINELPREPARE(Prepare)), Str(str), Num(num), File(file) { sentinelDeviceSend(&Base.Base, sizeof(stdio_fgets)); }
	char *Str;
	char *RC;
};

struct stdio_fputs {
	static __forceinline__ __device__ char *Prepare(stdio_fputs *t, char *data, char *dataEnd, intptr_t offset) {
		int strLength = t->Str ? (int)strlen(t->Str) + 1 : 0;
		char *str = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += strLength);
		if (end > dataEnd) return nullptr;
		memcpy(str, t->Str, strLength);
		if (t->Str) t->Str = str + offset;
		return end;
	}
	sentinelJumboMessage Base;
	const char *Str; FILE *File;
	__device__ stdio_fputs(bool wait, const char *str, FILE *file) : Base(STDIO_FPUTS, (wait ? FLOW_WAIT : FLOW_NONE) | FLOW_JUMBOOUT, SENTINEL_CHUNK, SENTINELPREPARE(Prepare)), Str(str), File(file) { sentinelDeviceSend(&Base.Base, sizeof(stdio_fputs)); }
	int RC;
};

struct stdio_ungetc {
	sentinelMessage Base;
	int Ch; FILE *File;
	__device__ stdio_ungetc(bool wait, int ch, FILE *file) : Base(STDIO_UNGETC, wait ? FLOW_WAIT : FLOW_NONE), Ch(ch), File(file) { sentinelDeviceSend(&Base, sizeof(stdio_ungetc)); }
	int RC;
};

struct stdio_fread {
	static __forceinline__ __device__ char *Prepare(stdio_fread *t, char *data, char *dataEnd, intptr_t offset) {
		char *ptr = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += SENTINEL_CHUNK);
		if (end > dataEnd) return nullptr;
		t->Base.Ptr = ptr + offset;
		t->Base.SafeSize = SENTINEL_CHUNK;
		t->Base.Size = t->Size * t->Num;
		return end;
	}
	static __forceinline__ __device__ bool Postfix(stdio_fread *t, intptr_t offset) {
		t->Base.Size = t->Size * t->RC;
		//char *ptr = (char *)t->Base.Ptr - offset;
		//if (t->RC > 0) memcpy(t->Buf, ptr, t->Size * t->RC);
		return true;
	}
	sentinelJumboMessage Base;
	void *Buf; size_t Size; size_t Num; FILE *File;
	__device__ stdio_fread(bool wait, void *buf, size_t size, size_t num, FILE *file) : Base(STDIO_FREAD, (wait ? FLOW_WAIT : FLOW_NONE) | FLOW_JUMBOIN, SENTINEL_CHUNK, SENTINELPREPARE(Prepare), SENTINELPOSTFIX(Postfix)), Buf(buf), Size(size), Num(num), File(file) { sentinelDeviceSend(&Base.Base, sizeof(stdio_fread)); }
	size_t RC;
};

struct stdio_fwrite {
	static __forceinline__ __device__ char *Prepare(stdio_fwrite *t, char *data, char *dataEnd, intptr_t offset) {
		size_t size = t->Size * t->Num;
		char *ptr = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += size);
		if (end > dataEnd) return nullptr;
		memcpy(ptr, t->Ptr, size);
		t->Ptr = ptr + offset;
		return end;
	}
	sentinelJumboMessage Base;
	const void *Ptr; size_t Size; size_t Num; FILE *File;
	__device__ stdio_fwrite(bool wait, const void *ptr, size_t size, size_t num, FILE *file) : Base(STDIO_FWRITE, (wait ? FLOW_WAIT : FLOW_NONE) | FLOW_JUMBOOUT, SENTINEL_CHUNK, SENTINELPREPARE(Prepare)), Ptr(ptr), Size(size), Num(num), File(file) { sentinelDeviceSend(&Base.Base, sizeof(stdio_fwrite)); }
	size_t RC;
};

struct stdio_fseek {
	sentinelMessage Base;
	FILE *File; long int Offset; int Origin;
	__device__ stdio_fseek(bool wait, FILE *file, long int offset, int origin) : Base(STDIO_FSEEK, wait ? FLOW_WAIT : FLOW_NONE), File(file), Offset(offset), Origin(origin) { sentinelDeviceSend(&Base, sizeof(stdio_fseek)); }
	int RC;
};

struct stdio_ftell {
	sentinelMessage Base;
	FILE *File;
	__device__ stdio_ftell(FILE *file) : Base(STDIO_FTELL, FLOW_WAIT), File(file) { sentinelDeviceSend(&Base, sizeof(stdio_ftell)); }
	int RC;
};

struct stdio_rewind {
	sentinelMessage Base;
	FILE *File;
	__device__ stdio_rewind(FILE *file) : Base(STDIO_REWIND, FLOW_WAIT), File(file) { sentinelDeviceSend(&Base, sizeof(stdio_rewind)); }
};

#if defined(__USE_LARGEFILE)
#ifndef __USE_LARGEFILE64
#define __off64_t char
#endif
struct stdio_fseeko {
	sentinelMessage Base;
	FILE *File; __off_t Offset; __off64_t Offset64; int Origin; bool Bit64;
	__device__ stdio_fseeko(bool wait, FILE *file, __off_t offset, __off64_t offset64, int origin, bool bit64) : Base(STDIO_FSEEKO, wait ? FLOW_WAIT : FLOW_NONE), File(file), Offset(offset), Offset64(offset64), Origin(origin), Bit64(bit64) { sentinelDeviceSend(&Base, sizeof(stdio_fseeko)); }
	int RC;
};

struct stdio_ftello {
	sentinelMessage Base;
	FILE *File; bool Bit64;
	__device__ stdio_ftello(FILE *file, bool bit64) : Base(STDIO_FTELLO, FLOW_WAIT), File(file), Bit64(bit64) { sentinelDeviceSend(&Base, sizeof(stdio_ftello)); }
	__off_t RC;
#if __USE_LARGEFILE64
	__off64_t RC64;
#endif
};
#endif

#ifndef __USE_LARGEFILE64
#define fpos64_t char
#endif
struct stdio_fgetpos {
	sentinelMessage Base;
	FILE *File; fpos_t *Pos; fpos64_t *Pos64; bool Bit64;
	__device__ stdio_fgetpos(FILE *__restrict file, fpos_t *__restrict pos, fpos64_t *__restrict pos64, bool bit64) : Base(STDIO_FGETPOS, FLOW_WAIT), File(file), Pos(pos), Pos64(pos64), Bit64(bit64) { sentinelDeviceSend(&Base, sizeof(stdio_fgetpos)); }
	int RC;
};

struct stdio_fsetpos {
	sentinelMessage Base;
	FILE *File; const fpos_t *Pos; const fpos64_t *Pos64; bool Bit64;
	__device__ stdio_fsetpos(FILE *__restrict file, const fpos_t *pos, const fpos64_t *pos64, bool bit64) : Base(STDIO_FSETPOS, FLOW_WAIT), File(file), Pos(pos), Pos64(pos64), Bit64(bit64) { sentinelDeviceSend(&Base, sizeof(stdio_fsetpos)); }
	int RC;
};

struct stdio_clearerr {
	sentinelMessage Base;
	FILE *File;
	__device__ stdio_clearerr(FILE *file) : Base(STDIO_CLEARERR, FLOW_NONE), File(file) { sentinelDeviceSend(&Base, sizeof(stdio_clearerr)); }
};

struct stdio_feof {
	sentinelMessage Base;
	FILE *File;
	__device__ stdio_feof(FILE *file) : Base(STDIO_FEOF, FLOW_WAIT), File(file) { sentinelDeviceSend(&Base, sizeof(stdio_feof)); }
	int RC;
};

struct stdio_ferror {
	sentinelMessage Base;
	FILE *File;
	__device__ stdio_ferror(FILE *file) : Base(STDIO_FERROR, FLOW_WAIT), File(file) { sentinelDeviceSend(&Base, sizeof(stdio_ferror)); }
	int RC;
};

struct stdio_fileno {
	sentinelMessage Base;
	FILE *File;
	__device__ stdio_fileno(FILE *file) : Base(STDIO_FILENO, FLOW_WAIT), File(file) { sentinelDeviceSend(&Base, sizeof(stdio_fileno)); }
	int RC;
};

#endif  /* _SENTINEL_STDIOMSG_H */