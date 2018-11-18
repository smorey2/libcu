/*
sentinel-timemsg.h - messages for sentinel
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
#ifndef _SENTINEL_TIMEMSG_H
#define _SENTINEL_TIMEMSG_H

#include <sentinel.h>
#include <crtdefscu.h>
#include <stringcu.h>

enum {
	TIME_TIME = 70,
	TIME_MKTIME,
	TIME_STRFTIME,
};

struct time_time {
	sentinelMessage Base;
	__device__ time_time() : Base(TIME_TIME, SENTINELFLOW_WAIT) { sentinelDeviceSend(&Base, sizeof(time_time)); }
	time_t RC;
};

struct time_mktime {
	sentinelMessage Base;
	struct tm *Tp;
	__device__ time_mktime(struct tm *tp) : Base(TIME_MKTIME, SENTINELFLOW_WAIT), Tp(tp) { sentinelDeviceSend(&Base, sizeof(time_mktime)); }
	time_t RC;
};

struct time_strftime {
	sentinelMessage Base;
	const char *Buf; size_t Maxsize; const char *Str; const struct tm Tp;
	__device__ time_strftime(const char *buf, size_t maxsize, const char *str, const struct tm *tp) : Base(TIME_STRFTIME, SENTINELFLOW_WAIT, SENTINEL_CHUNK), Buf(buf), Maxsize(maxsize), Str(str), Tp(*tp) { sentinelDeviceSend(&Base, sizeof(time_strftime), PtrsIn, PtrsOut); }
	size_t RC;
	void *Ptr;
	sentinelInPtr PtrsIn[2] = {
		{ &Str, -1 },
		nullptr
	};
	sentinelOutPtr PtrsOut[3] = {
		{ (void *)-1 },
		{ &Ptr, &Buf, -1, &RC },
		nullptr
	};
};

#endif  /* _SENTINEL_TIMEMSG_H */