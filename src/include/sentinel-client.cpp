#include <sentinel.h>
#include <sentinel-hostmsg.h>
#if __OS_WIN
#include <windows.h>
#define HOST_SPINLOCK(DELAY, CMP, SET) while (_InterlockedCompareExchange(control, SET, CMP) != CMP) { Sleep(DELAY); }
#elif __OS_UNIX
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#define HOST_SPINLOCK(DELAY, CMP, SET) while (__sync_val_compare_and_swap((volatile long *)control, SET, CMP) != CMP) { sleep(DELAY); }
#endif
#include <stdio.h>

#if HAS_HOSTSENTINEL

static void executeTrans(sentinelCommand *cmd, int size, sentinelInPtr *listIn, sentinelOutPtr *listOut, intptr_t offset, char *&trans);

static char *preparePtrs(sentinelInPtr *ptrsIn, sentinelOutPtr *ptrsOut, sentinelCommand *cmd, char *data, char *dataEnd, intptr_t offset, sentinelOutPtr *&listOut, char *&trans) {
	char *ptr = data, *next;

	// PREPARE & TRANSFER
	int transSize = 0;
	sentinelInPtr *listIn = nullptr;
	if (ptrsIn)
		for (sentinelInPtr *p = ptrsIn; p->field; p++) {
			char **field = (char **)p->field;
			int size = p->size != -1 ? p->size : (p->size = *field ? (int)strlen(*field) + 1 : 0);
			next = ptr + size;
			if (!size) p->unknown = nullptr;
			else if (next <= dataEnd) { p->unknown = ptr; ptr = next; }
			else { p->unknown = listIn; listIn = p; transSize += size; }
		}
	if (ptrsOut) {
		sentinelOutPtr *listOut_ = nullptr;
		ptr = ptrsOut[0].field == (char *)-1 ? ptr : data;
		for (sentinelOutPtr *p = ptrsOut; p->field; p++) {
			char **field = (char **)p->field;
			int size = p->size != -1 ? p->size : (int)(dataEnd - ptr);
			next = ptr + size;
			if (!size) {}
			else if (next <= dataEnd) { *field = ptr + offset; ptr = next; }
			else { p->unknown = listOut_; listOut_ = p; transSize += size; }
		}
		listOut = listOut_;
	}
	if (transSize) executeTrans(cmd, transSize, listIn, nullptr, offset, trans);

	// PACK
	for (sentinelInPtr *p = ptrsIn; p->field; p++) {
		char **field = (char **)p->field;
		char *ptr = (char *)p->unknown;
		if (!ptr || !*field)
			continue;
		memcpy(ptr, *field, p->size);
		*field = ptr + offset;
	}
	return data;
}

static bool postfixPtrs(sentinelOutPtr *ptrsOut, sentinelCommand *cmd, intptr_t offset) {
	// UNPACK
	for (sentinelOutPtr *p = ptrsOut; p->field; p++) {
		char **buf = (char **)p->buf;
		if (!*buf)
			continue;
		char **field = (char **)p->field;
		char *ptr = *field - offset;
		int *sizeField = (int *)p->sizeField;
		int size = !*sizeField ? p->size : *sizeField;
		if (size > 0) memcpy(*buf, ptr, size);
	}
	return true;
}

static sentinelMap *_sentinelClientMap = nullptr;
static intptr_t _sentinelClientMapOffset = 0;
void sentinelClientSend(sentinelMessage *msg, int msgLength, sentinelInPtr *ptrsIn, sentinelOutPtr *ptrsOut) {
#ifndef _WIN64
	printf("Sentinel client currently only works in x64.\n"); abort();
#else
	sentinelMap *map = _sentinelClientMap;
	if (!map)
		panic("sentinel: client map not defined. did you start sentinel?\n");

	// ATTACH
#if __OS_WIN
	long id = _InterlockedAdd(&map->setId, SENTINEL_MSGSIZE) - SENTINEL_MSGSIZE;
#elif __OS_UNIX
	long id = __sync_fetch_and_add((volatile long *)&map->setId, SENTINEL_MSGSIZE) - SENTINEL_MSGSIZE;
#endif
	sentinelCommand *cmd = (sentinelCommand *)&map->data[id % sizeof(map->data)];
	if (cmd->magic != SENTINEL_MAGIC)
		panic("bad sentinel magic");
	volatile long *control = &cmd->control; char *trans = nullptr; intptr_t offset = _sentinelClientMapOffset;
	HOST_SPINLOCK(25, SENTINELCONTROL_NORMAL, SENTINELCONTROL_DEVICE);

	// PREPARE
	char *data = cmd->data + ROUND8_(msgLength), *dataEnd = data + msg->size;
	sentinelOutPtr *listOut = nullptr;
	if (((ptrsIn || ptrsOut) && !(data = preparePtrs(ptrsIn, ptrsOut, cmd, data, dataEnd, offset, listOut, trans))) ||
		(msg->prepare && !msg->prepare(msg, data, dataEnd, offset)))
		panic("msg too long");
	cmd->length = msgLength; memcpy(cmd->data, msg, msgLength);
	//printf("msg: %d[%d]'", msg->op, msgLength); for (int i = 0; i < msgLength; i++) printf("%02x", ((char *)msg)[i] & 0xff); printf("'\n");
	*control = SENTINELCONTROL_DEVICERDY;

	// FLOW-WAIT
	if (msg->flow & SENTINELFLOW_WAIT) {
		HOST_SPINLOCK(25, SENTINELCONTROL_HOSTRDY, SENTINELCONTROL_DEVICEWAIT);
		if (listOut) executeTrans(cmd, 0, nullptr, listOut, offset, trans);
		cmd->length = msgLength; memcpy(msg, cmd->data, msgLength);
		if ((ptrsOut && !postfixPtrs(ptrsOut, cmd, offset)) ||
			(msg->postfix && !msg->postfix(msg, offset)))
			panic("postfix error");
		*control = SENTINELCONTROL_DEVICEDONE;
	}
#endif
}

static void executeTrans(sentinelCommand *cmd, int size, sentinelInPtr *listIn, sentinelOutPtr *listOut, intptr_t offset, char *&trans) {
	volatile long *control = &cmd->control;
	char *data = cmd->data, *ptr = trans;
	// create memory
	if (size) {
		*(int *)data = size;
		*control = SENTINELCONTROL_TRANSSIZE;
		HOST_SPINLOCK(25, SENTINELCONTROL_TRANRDY, SENTINELCONTROL_TRANDONE);
		ptr = trans = *(char **)data;
	}
	// transfer
	if (listIn) {
		for (sentinelInPtr *p = listIn; p; p = (sentinelInPtr *)p->unknown) {
			char **field = (char **)p->field;
			const char *v = (const char *)*field; int size = p->size, remain = size, length = 0;
			while (remain > 0) {
				length = cmd->length = remain > SENTINEL_MSGSIZE ? SENTINEL_MSGSIZE : remain;
				memcpy(data, (void *)v, length); remain -= length; v += length;
				*control = SENTINELCONTROL_TRANSIN;
				HOST_SPINLOCK(25, SENTINELCONTROL_TRANRDY, SENTINELCONTROL_TRANDONE);
			}
			*field = ptr; ptr += size;
			p->unknown = nullptr;
		}
	}
	if (listOut) {
		for (sentinelInPtr *p = listIn; p; p = (sentinelInPtr *)p->unknown) {
			char **field = (char **)p->field;
			const char *v = (const char *)*field; int size = p->size, remain = size, length = 0;
			while (remain > 0) {
				length = cmd->length = remain > SENTINEL_MSGSIZE ? SENTINEL_MSGSIZE : remain;
				memcpy(data, (void *)v, length); remain -= length; v += length;
				*control = SENTINELCONTROL_TRANSOUT;
				HOST_SPINLOCK(25, SENTINELCONTROL_TRANRDY, SENTINELCONTROL_TRANDONE);
			}
			*field = ptr; ptr += size;
			p->unknown = nullptr;
		}
	}
}

#if __OS_WIN
static HANDLE _clientMapHandle = NULL;
static int *_clientMap = nullptr;
#elif __OS_UNIX
static void *_clientMap = nullptr;
#endif

void sentinelClientInitialize(char *mapHostName) {
#if __OS_WIN
	_clientMapHandle = OpenFileMapping(FILE_MAP_ALL_ACCESS, FALSE, mapHostName);
	if (!_clientMapHandle) {
		printf("(%d) Could not connect to Sentinel host. Please ensure host application is running.\n", GetLastError()); exit(1);
	}
	_clientMap = (int *)MapViewOfFile(_clientMapHandle, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(sentinelMap) + MEMORY_ALIGNMENT);
	if (!_clientMap) {
		printf("(%d) Could not map view of file.\n", GetLastError());
		CloseHandle(_clientMapHandle); exit(1);
	}
	_sentinelClientMap = (sentinelMap *)ROUNDN_(_clientMap, MEMORY_ALIGNMENT);
	_sentinelClientMapOffset = (intptr_t)((char *)_sentinelClientMap->offset - (char *)_sentinelClientMap);
#elif __OS_UNIX
	struct stat sb;
	int fd = open(mapHostName, O_RDONLY);
	if (fd == -1) { perror("open"); exit(1); }
	if (fstat(fd, &sb) == -1) { perror("fstat"); exit(1); }
	if (!S_ISREG(sb.st_mode)) { fprintf(stderr, "%s is not a file\n", mapHostName); exit(1); }
	_clientMap = mmap(NULL, sizeof(sentinelMap) + MEMORY_ALIGNMENT, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, fd, 0);
	if (!_clientMap) {
		printf("Could not connect to Sentinel host. Please ensure host application is running.\n"); exit(1);
	}
	if (close(fd) == -1) { perror("close"); exit(1); }
	_sentinelClientMap = (sentinelMap *)ROUNDN_(_clientMap, MEMORY_ALIGNMENT);
	_sentinelClientMapOffset = 0;
#endif
}

void sentinelClientShutdown() {
#if __OS_WIN
	if (_clientMap) { UnmapViewOfFile(_clientMap); _clientMap = nullptr; }
	if (_clientMapHandle) { CloseHandle(_clientMapHandle); _clientMapHandle = NULL; }
#elif __OS_UNIX
	if (_clientMap) { munmap(_clientMap, sizeof(sentinelMap) + MEMORY_ALIGNMENT); _clientMap = nullptr; }
#endif
}

static __forceinline__ int getprocessid_() { host_getprocessid msg; return msg.rc; }

static char *sentinelClientRedirPipelineArgs[] = { (char *)"^0" };
void sentinelClientRedir(pipelineRedir *redir) {
#if __OS_WIN
	HANDLE process = OpenProcess(PROCESS_DUP_HANDLE, FALSE, getprocessid_());
	pipelineCreate(1, sentinelClientRedirPipelineArgs, nullptr, &redir[1].input, &redir[1].output, &redir[1].error, process, redir);
#elif __OS_UNIX
#endif
}

#endif