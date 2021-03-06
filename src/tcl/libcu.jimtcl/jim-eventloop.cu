#pragma region License
/* Jim - A small embeddable Tcl interpreter
*
* Copyright 2005 Salvatore Sanfilippo <antirez@invece.org>
* Copyright 2005 Clemens Hintze <c.hintze@gmx.net>
* Copyright 2005 patthoyts - Pat Thoyts <patthoyts@users.sf.net>
* Copyright 2008 oharboe - Øyvind Harboe - oyvind.harboe@zylin.com
* Copyright 2008 Andrew Lunn <andrew@lunn.ch>
* Copyright 2008 Duane Ellis <openocd@duaneellis.com>
* Copyright 2008 Uwe Klein <uklein@klein-messgeraete.de>
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*
* 1. Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above
*    copyright notice, this list of conditions and the following
*    disclaimer in the documentation and/or other materials
*    provided with the distribution.
*
* THIS SOFTWARE IS PROVIDED BY THE JIM TCL PROJECT ``AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
* THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
* PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
* JIM TCL PROJECT OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
* INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
* OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
* HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
* STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
* ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
* ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* The views and conclusions contained in the software and documentation
* are those of the authors and should not be interpreted as representing
* official policies, either expressed or implied, of the Jim Tcl Project.
**/
#pragma endregion

#include "jimautoconf.h"
#include "jim.h"
#include "jim-eventloop.h"
// POSIX includes
#include <timecu.h>
//#include <sys/time.h>
//#include <sys/types.h>
#include <stringcu.h>
#include <unistdcu.h>
//#include <errno.h>
#ifdef __CUDA_ARCH__
#define msleep(MS) sleep((MS) / 1000)
#elif defined(__MINGW32__)
#include <windows.h>
#include <winsock.h>
#define msleep Sleep
#else
#ifdef HAVE_SYS_SELECT_H
#include <sys/select.h>
#endif

#ifndef HAVE_USLEEP
// XXX: Implement this in terms of select() or nanosleep()
#define msleep(MS) sleep((MS) / 1000)
//#warning "sub-second sleep not supported"
#else
#define msleep(MS) sleep((MS) / 1000); usleep(((MS) % 1000) * 1000);
#endif
#endif

// ---

// File event structure
typedef struct Jim_FileEvent {
	FILE *handle;
	int mask;                   // one of JIM_EVENT_(READABLE|WRITABLE|EXCEPTION)
	Jim_FileProc *fileProc;
	Jim_EventFinalizerProc *finalizerProc;
	void *clientData;
	struct Jim_FileEvent *next;
} Jim_FileEvent;

// Time event structure
typedef struct Jim_TimeEvent {
	jim_wide id;                // time event identifier
	long initialms;             // initial relative timer value
	jim_wide when;              // milliseconds
	Jim_TimeProc *timeProc;
	Jim_EventFinalizerProc *finalizerProc;
	void *clientData;
	struct Jim_TimeEvent *next;
} Jim_TimeEvent;

// Per-interp stucture containing the state of the event loop
typedef struct Jim_EventLoop {
	Jim_FileEvent *fileEventHead;
	Jim_TimeEvent *timeEventHead;
	jim_wide timeEventNextId;   // highest event id created, starting at 1
	time_t timeBase;
	int suppress_bgerror; // bgerror returned break, so don't call it again
} Jim_EventLoop;

static __device__ void JimAfterTimeHandler(Jim_Interp *interp, void *clientData);
static __device__ void JimAfterTimeEventFinalizer(Jim_Interp *interp, void *clientData);

__device__ int Jim_EvalObjBackground(Jim_Interp *interp, Jim_Obj *scriptObjPtr) {
	Jim_EventLoop *eventLoop = (Jim_EventLoop *)Jim_GetAssocData(interp, "eventloop");
	Jim_CallFrame *savedFramePtr = interp->framePtr;
	interp->framePtr = interp->topFramePtr;
	int retval = Jim_EvalObj(interp, scriptObjPtr);
	interp->framePtr = savedFramePtr;
	// Try to report the error (if any) via the bgerror proc
	if (retval != JIM_OK && !eventLoop->suppress_bgerror)
		Jim_BackgroundError(interp);
	return retval;
}

__device__ void Jim_BackgroundError(Jim_Interp *interp) {
	Jim_EventLoop *eventLoop = (Jim_EventLoop *)Jim_GetAssocData(interp, "eventloop");
	// Try to report the error (if any) via the bgerror proc
	if (!eventLoop->suppress_bgerror) {
		int rc = JIM_ERROR;
		Jim_Obj *objv[2];
		objv[0] = Jim_NewStringObj(interp, "bgerror", -1);
		objv[1] = Jim_GetResult(interp);
		Jim_IncrRefCount(objv[0]);
		Jim_IncrRefCount(objv[1]);
		if (Jim_GetCommand(interp, objv[0], JIM_NONE) == NULL || (rc = Jim_EvalObjVector(interp, 2, objv)) != JIM_OK) {
			// No more bgerror calls
			if (rc == JIM_BREAK)
				eventLoop->suppress_bgerror++;
			else {
				// Report the error to stderr
				Jim_MakeErrorMessage(interp);
				fprintf_(stderr, "%s\n", Jim_String(Jim_GetResult(interp)));
				// And reset the result
				Jim_SetResultString(interp, "", -1);
			}
		}
		Jim_DecrRefCount(interp, objv[0]);
		Jim_DecrRefCount(interp, objv[1]);
	}
}

__device__ void Jim_CreateFileHandler(Jim_Interp *interp, FILE *handle, int mask, Jim_FileProc *proc, void *clientData, Jim_EventFinalizerProc *finalizerProc) {
	Jim_EventLoop *eventLoop = (Jim_EventLoop *)Jim_GetAssocData(interp, "eventloop");
	Jim_FileEvent *fe = (Jim_FileEvent *)Jim_Alloc(sizeof(*fe));
	fe->handle = handle;
	fe->mask = mask;
	fe->fileProc = proc;
	fe->finalizerProc = finalizerProc;
	fe->clientData = clientData;
	fe->next = eventLoop->fileEventHead;
	eventLoop->fileEventHead = fe;
}

// Removes all event handlers for 'handle' that match 'mask'.
__device__ void Jim_DeleteFileHandler(Jim_Interp *interp, FILE *handle, int mask) {
	Jim_FileEvent *next, *prev = NULL;
	Jim_EventLoop *eventLoop = (Jim_EventLoop *)Jim_GetAssocData(interp, "eventloop");
	for (Jim_FileEvent *fe = eventLoop->fileEventHead; fe; fe = next) {
		next = fe->next;
		if (fe->handle == handle && (fe->mask & mask)) {
			// Remove this entry from the list
			if (prev == NULL)
				eventLoop->fileEventHead = next;
			else
				prev->next = next;
			if (fe->finalizerProc)
				fe->finalizerProc(interp, fe->clientData);
			Jim_Free(fe);
			continue;
		}
		prev = fe;
	}
}

// Returns the time since interp creation in milliseconds.
static __device__ jim_wide JimGetTime(Jim_EventLoop *eventLoop) {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (jim_wide)(tv.tv_sec - eventLoop->timeBase) * 1000 + tv.tv_usec / 1000;
}

__device__ jim_wide Jim_CreateTimeHandler(Jim_Interp *interp, jim_wide milliseconds, Jim_TimeProc *proc, void *clientData, Jim_EventFinalizerProc *finalizerProc) {
	Jim_EventLoop *eventLoop = (Jim_EventLoop *)Jim_GetAssocData(interp, "eventloop");
	jim_wide id = ++eventLoop->timeEventNextId;
	Jim_TimeEvent *te = (Jim_TimeEvent *)Jim_Alloc(sizeof(*te));
	te->id = id;
	te->initialms = (long)milliseconds;
	te->when = JimGetTime(eventLoop) + milliseconds;
	te->timeProc = proc;
	te->finalizerProc = finalizerProc;
	te->clientData = clientData;
	// Add to the appropriate place in the list
	Jim_TimeEvent *prev = NULL;
	for (Jim_TimeEvent *e = eventLoop->timeEventHead; e; e = e->next) {
		if (te->when < e->when)
			break;
		prev = e;
	}
	if (prev) {
		te->next = prev->next;
		prev->next = te;
	}
	else {
		te->next = eventLoop->timeEventHead;
		eventLoop->timeEventHead = te;
	}
	return id;
}

static __device__ jim_wide JimParseAfterId(Jim_Obj *idObj) {
	const char *tok = Jim_String(idObj);
	jim_wide id;
	return (!strncmp(tok, "after#", 6) && Jim_StringToWide(tok + 6, &id, 10) == JIM_OK ? id : -1); // Got an event by id
}

static __device__ jim_wide JimFindAfterByScript(Jim_EventLoop *eventLoop, Jim_Obj *scriptObj) {
	for (Jim_TimeEvent *te = eventLoop->timeEventHead; te; te = te->next)
		if (te->timeProc == JimAfterTimeHandler) // Is this an 'after' event?
			if (Jim_StringEqObj(scriptObj, (Jim_Obj *)te->clientData))
				return te->id;
	return -1; // NO event with the specified ID found
}

static __device__ Jim_TimeEvent *JimFindTimeHandlerById(Jim_EventLoop *eventLoop, jim_wide id) {
	for (Jim_TimeEvent *te = eventLoop->timeEventHead; te; te = te->next)
		if (te->id == id)
			return te;
	return NULL;
}

static __device__ Jim_TimeEvent *Jim_RemoveTimeHandler(Jim_EventLoop *eventLoop, jim_wide id) {
	Jim_TimeEvent *prev = NULL;
	for (Jim_TimeEvent *te = eventLoop->timeEventHead; te; te = te->next) {
		if (te->id == id) {
			if (prev == NULL)
				eventLoop->timeEventHead = te->next;
			else
				prev->next = te->next;
			return te;
		}
		prev = te;
	}
	return NULL;
}

static __device__ void Jim_FreeTimeHandler(Jim_Interp *interp, Jim_TimeEvent *te) {
	if (te->finalizerProc)
		te->finalizerProc(interp, te->clientData);
	Jim_Free(te);
}

__device__ jim_wide Jim_DeleteTimeHandler(Jim_Interp *interp, jim_wide id) {
	Jim_EventLoop *eventLoop = (Jim_EventLoop *)Jim_GetAssocData(interp, "eventloop");
	if (id > eventLoop->timeEventNextId)
		return -2; // wrong event ID
	Jim_TimeEvent *te = Jim_RemoveTimeHandler(eventLoop, id);
	if (te) {
		jim_wide remain = te->when - JimGetTime(eventLoop);
		remain = (remain < 0 ? 0 : remain);
		Jim_FreeTimeHandler(interp, te);
		return remain;
	}
	return -1; // NO event with the specified ID found
}

// --- POSIX version of Jim_ProcessEvents, for now the only available ---
#pragma region POSIX Jim_ProcessEvents

// Process every pending time event, then every pending file event (that may be registered by time event callbacks just processed).
// The behaviour depends upon the setting of flags:
// If flags is 0, the function does nothing and returns.
// if flags has JIM_ALL_EVENTS set, all event types are processed.
// if flags has JIM_FILE_EVENTS set, file events are processed.
// if flags has JIM_TIME_EVENTS set, time events are processed.
// if flags has JIM_DONT_WAIT set, the function returns as soon as all the events that are possible to process without waiting are processed.
// Returns the number of events processed or -1 if there are no matching handlers, or -2 on error.
__device__ int Jim_ProcessEvents(Jim_Interp *interp, int flags) {
	jim_wide sleep_ms = -1;
	int processed = 0;
	Jim_EventLoop *eventLoop = (Jim_EventLoop *)Jim_GetAssocData(interp, "eventloop");
	Jim_FileEvent *fe = eventLoop->fileEventHead;
	if ((flags & JIM_FILE_EVENTS) == 0 || fe == NULL)
		if ((flags & JIM_TIME_EVENTS) == 0 || eventLoop->timeEventHead == NULL) // No file events
			return -1; // No time events
	// Note that we want call select() even if there are no file events to process as long as we want to process time events, in order to sleep until the next time event is ready to fire.
	if (flags & JIM_DONT_WAIT)
		sleep_ms = 0; // Wait no time
	else if (flags & JIM_TIME_EVENTS) {
		// The nearest timer is always at the head of the list
		if (eventLoop->timeEventHead) {
			Jim_TimeEvent *shortest = eventLoop->timeEventHead;
			// Calculate the time missing for the nearest timer to fire
			sleep_ms = shortest->when - JimGetTime(eventLoop);
			if (sleep_ms < 0)
				sleep_ms = 0;
		}
		else
			sleep_ms = -1; // Wait forever
	}

#ifdef HAVE_SELECT
	if (flags & JIM_FILE_EVENTS) {
		fd_set rfds, wfds, efds;
		FD_ZERO(&rfds);
		FD_ZERO(&wfds);
		FD_ZERO(&efds);
		// Check file events
		int maxfd = -1;
		while (fe != NULL) {
			int fd = fileno(fe->handle);
			if (fe->mask & JIM_EVENT_READABLE) FD_SET(fd, &rfds);
			if (fe->mask & JIM_EVENT_WRITABLE) FD_SET(fd, &wfds);
			if (fe->mask & JIM_EVENT_EXCEPTION) FD_SET(fd, &efds);
			if (maxfd < fd)
				maxfd = fd;
			fe = fe->next;
		}
		struct timeval tv, *tvp = NULL;
		if (sleep_ms >= 0) {
			tvp = &tv;
			tvp->tv_sec = sleep_ms / 1000;
			tvp->tv_usec = 1000 * (sleep_ms % 1000);
		}
		int retval = select(maxfd + 1, &rfds, &wfds, &efds, tvp);
		if (retval < 0) {
			if (errno == EINVAL) {
				// This can happen on mingw32 if a non-socket filehandle is passed
				Jim_SetResultString(interp, "non-waitable filehandle", -1);
				return -2;
			}
		}
		else if (retval > 0) {
			fe = eventLoop->fileEventHead;
			while (fe != NULL) {
				int fd = fileno(fe->handle);
				int mask = 0;
				if ((fe->mask & JIM_EVENT_READABLE) && FD_ISSET(fd, &rfds)) mask |= JIM_EVENT_READABLE;
				if (fe->mask & JIM_EVENT_WRITABLE && FD_ISSET(fd, &wfds)) mask |= JIM_EVENT_WRITABLE;
				if (fe->mask & JIM_EVENT_EXCEPTION && FD_ISSET(fd, &efds)) mask |= JIM_EVENT_EXCEPTION;
				if (mask) {
					if (fe->fileProc(interp, fe->clientData, mask) != JIM_OK)
						Jim_DeleteFileHandler(interp, fe->handle, mask); // Remove the element on handler error
					processed++;
					// After an event is processed our file event list may no longer be the same, so what we do is to clear the bit for this file descriptor and restart again from the head.
					fe = eventLoop->fileEventHead;
					FD_CLR(fd, &rfds);
					FD_CLR(fd, &wfds);
					FD_CLR(fd, &efds);
				}
				else
					fe = fe->next;
			}
		}
	}
#else
	if (sleep_ms > 0)
		msleep((unsigned long)sleep_ms);
#endif

	// Check time events
	Jim_TimeEvent *te = eventLoop->timeEventHead;
	jim_wide maxId = eventLoop->timeEventNextId;
	while (te) {
		if (te->id > maxId) {
			te = te->next;
			continue;
		}
		if (JimGetTime(eventLoop) >= te->when) {
			jim_wide id = te->id;
			// Remove from the list before executing
			Jim_RemoveTimeHandler(eventLoop, id);
			te->timeProc(interp, te->clientData);
			// After an event is processed our time event list may no longer be the same, so we restart from head.
			// Still we make sure to don't process events registered by event handlers itself in order to don't loop forever
			// even in case an [after 0] that continuously register itself. To do so we saved the max ID we want to handle.
			Jim_FreeTimeHandler(interp, te);
			te = eventLoop->timeEventHead;
			processed++;
		}
		else
			te = te->next;
	}
	return processed;
}

#pragma endregion

static __device__ void JimELAssocDataDeleProc(Jim_Interp *interp, void *data) {
	void *next;
	Jim_EventLoop *eventLoop = (Jim_EventLoop *)data;
	Jim_FileEvent *fe = eventLoop->fileEventHead;
	while (fe) {
		next = fe->next;
		if (fe->finalizerProc)
			fe->finalizerProc(interp, fe->clientData);
		Jim_Free(fe);
		fe = (Jim_FileEvent *)next;
	}
	Jim_TimeEvent *te = eventLoop->timeEventHead;
	while (te) {
		next = te->next;
		if (te->finalizerProc)
			te->finalizerProc(interp, te->clientData);
		Jim_Free(te);
		te = (Jim_TimeEvent *)next;
	}
	Jim_Free(data);
}

static __device__ int JimELVwaitCommand(ClientData dummy, Jim_Interp *interp, int argc, Jim_Obj *const *argv) {
	if (argc != 2) {
		Jim_WrongNumArgs(interp, 1, argv, "name");
		return JIM_ERROR;
	}
	Jim_EventLoop *eventLoop = (Jim_EventLoop *)Jim_CmdPrivData(interp);
	Jim_Obj *oldValue = Jim_GetVariable(interp, argv[1], JIMGLOBAL_);
	if (oldValue)
		Jim_IncrRefCount(oldValue);
	// If a result was left, it is an error
	else if (Jim_Length(Jim_GetResult(interp)))
		return JIM_ERROR;
	eventLoop->suppress_bgerror = 0;
	int rc;
	while ((rc = Jim_ProcessEvents(interp, JIM_ALL_EVENTS)) >= 0) {
		Jim_Obj *currValue;
		currValue = Jim_GetVariable(interp, argv[1], JIMGLOBAL_);
		// Stop the loop if the vwait-ed variable changed value, or if was unset and now is set (or the contrary) or if a signal was caught
		if ((oldValue && !currValue) || (!oldValue && currValue) || (oldValue && currValue && !Jim_StringEqObj(oldValue, currValue)) || Jim_CheckSignal(interp))
			break;
	}
	if (oldValue)
		Jim_DecrRefCount(interp, oldValue);
	if (rc == -2)
		return JIM_ERROR;
	Jim_ResetResult(interp);
	return JIM_OK;
}

__constant__ static const char *const JimELUpdateCommand_options[] = {
	"idletasks", NULL
};
static __device__ int JimELUpdateCommand(ClientData dummy, Jim_Interp *interp, int argc, Jim_Obj *const *argv) {
	enum { UPDATE_IDLE, UPDATE_NONE };
	Jim_EventLoop *eventLoop = (Jim_EventLoop *)Jim_CmdPrivData(interp);
	int option = UPDATE_NONE;
	int flags = JIM_TIME_EVENTS;
	if (argc == 1)
		flags = JIM_ALL_EVENTS;
	else if (argc > 2 || Jim_GetEnum(interp, argv[1], JimELUpdateCommand_options, &option, NULL, JIM_ERRMSG | JIM_ENUM_ABBREV) != JIM_OK) {
		Jim_WrongNumArgs(interp, 1, argv, "?idletasks?");
		return JIM_ERROR;
	}
	eventLoop->suppress_bgerror = 0;
	while (Jim_ProcessEvents(interp, flags | JIM_DONT_WAIT) > 0) {}
	return JIM_OK;
}

static __device__ void JimAfterTimeHandler(Jim_Interp *interp, void *clientData) {
	Jim_Obj *objPtr = (Jim_Obj *)clientData;
	Jim_EvalObjBackground(interp, objPtr);
}

static __device__ void JimAfterTimeEventFinalizer(Jim_Interp *interp, void *clientData) {
	Jim_Obj *objPtr = (Jim_Obj *)clientData;
	Jim_DecrRefCount(interp, objPtr);
}

__constant__ static const char *const JimELAfterCommand_options[] = {
	"cancel", "info", "idle", NULL
};

static __device__ int JimELAfterCommand(ClientData dummy, Jim_Interp *interp, int argc, Jim_Obj *const *argv) {
	enum { AFTER_CANCEL, AFTER_INFO, AFTER_IDLE, AFTER_RESTART, AFTER_EXPIRE, AFTER_CREATE };
	if (argc < 2) {
		Jim_WrongNumArgs(interp, 1, argv, "option ?arg ...?");
		return JIM_ERROR;
	}
	Jim_EventLoop *eventLoop = (Jim_EventLoop *)Jim_CmdPrivData(interp);
	jim_wide ms = 0, id;
	int option = AFTER_CREATE;
	if (Jim_GetWide(interp, argv[1], &ms) != JIM_OK) {
		if (Jim_GetEnum(interp, argv[1], JimELAfterCommand_options, &option, "argument", JIM_ERRMSG) != JIM_OK)
			return JIM_ERROR;
		Jim_ResetResult(interp);
	}
	else if (argc == 2) {
		// Simply a sleep
		msleep((unsigned long)ms);
		return JIM_OK;
	}
	switch (option) {
	case AFTER_IDLE:
		if (argc < 3) {
			Jim_WrongNumArgs(interp, 2, argv, "script ?script ...?");
			return JIM_ERROR;
		}
		// fall through
	case AFTER_CREATE: {
		Jim_Obj *scriptObj = Jim_ConcatObj(interp, argc - 2, argv + 2);
		Jim_IncrRefCount(scriptObj);
		id = Jim_CreateTimeHandler(interp, ms, JimAfterTimeHandler, scriptObj, JimAfterTimeEventFinalizer);
		Jim_Obj *objPtr = Jim_NewStringObj(interp, NULL, 0);
		Jim_AppendString(interp, objPtr, "after#", -1);
		Jim_Obj *idObjPtr = Jim_NewIntObj(interp, id);
		Jim_IncrRefCount(idObjPtr);
		Jim_AppendObj(interp, objPtr, idObjPtr);
		Jim_DecrRefCount(interp, idObjPtr);
		Jim_SetResult(interp, objPtr);
		return JIM_OK; }
	case AFTER_CANCEL:
		if (argc < 3) {
			Jim_WrongNumArgs(interp, 2, argv, "id|command");
			return JIM_ERROR;
		}
		else {
			id = JimParseAfterId(argv[2]);
			if (id <= 0) {
				// Not an event id, so search by script
				Jim_Obj *scriptObj = Jim_ConcatObj(interp, argc - 2, argv + 2);
				id = JimFindAfterByScript(eventLoop, scriptObj);
				Jim_FreeNewObj(interp, scriptObj);
				if (id <= 0)
					break; // Not found
			}
			jim_wide remain = Jim_DeleteTimeHandler(interp, id);
			if (remain >= 0)
				Jim_SetResultInt(interp, remain);
		}
		break;
	case AFTER_INFO:
		if (argc == 2) {
			Jim_TimeEvent *te = eventLoop->timeEventHead;
			Jim_Obj *listObj = Jim_NewListObj(interp, NULL, 0);
			char buf[30];
			const char *fmt = "after#%" JIM_WIDE_MODIFIER;
			while (te) {
				snprintf(buf, sizeof(buf), fmt, te->id);
				Jim_ListAppendElement(interp, listObj, Jim_NewStringObj(interp, buf, -1));
				te = te->next;
			}
			Jim_SetResult(interp, listObj);
		}
		else if (argc == 3) {
			id = JimParseAfterId(argv[2]);
			if (id >= 0) {
				Jim_TimeEvent *e = JimFindTimeHandlerById(eventLoop, id);
				if (e && e->timeProc == JimAfterTimeHandler) {
					Jim_Obj *listObj = Jim_NewListObj(interp, NULL, 0);
					Jim_ListAppendElement(interp, listObj, (Jim_Obj *)e->clientData);
					Jim_ListAppendElement(interp, listObj, Jim_NewStringObj(interp, e->initialms ? "timer" : "idle", -1));
					Jim_SetResult(interp, listObj);
					return JIM_OK;
				}
			}
			Jim_SetResultFormatted(interp, "event \"%#s\" doesn't exist", argv[2]);
			return JIM_ERROR;
		}
		else {
			Jim_WrongNumArgs(interp, 2, argv, "?id?");
			return JIM_ERROR;
		}
		break;
	}
	return JIM_OK;
}

__device__ int Jim_eventloopInit(Jim_Interp *interp) {
	if (Jim_PackageProvide(interp, "eventloop", "1.0", JIM_ERRMSG))
		return JIM_ERROR;
	Jim_EventLoop *eventLoop = (Jim_EventLoop *)Jim_Alloc(sizeof(*eventLoop));
	memset(eventLoop, 0, sizeof(*eventLoop));
	Jim_SetAssocData(interp, "eventloop", JimELAssocDataDeleProc, eventLoop);
	Jim_CreateCommand(interp, "vwait", JimELVwaitCommand, eventLoop, NULL);
	Jim_CreateCommand(interp, "update", JimELUpdateCommand, eventLoop, NULL);
	Jim_CreateCommand(interp, "after", JimELAfterCommand, eventLoop, NULL);
	return JIM_OK;
}
