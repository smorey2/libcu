#include <sentinel.h>
#include <sentinel-hostmsg.h>
#if __OS_WIN
#include <windows.h>
#define HOST_SPINLOCK(SET, WHEN) while (InterlockedCompareExchange((long *)control, SET, WHEN) != WHEN) {}
#elif __OS_UNIX
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#define HOST_SPINLOCK(SET, WHEN) while (__sync_val_compare_and_swap((long *)control, SET, WHEN) != WHEN) {}
#endif
#include <stdio.h>

#if HAS_HOSTSENTINEL

static char *preparePtrs(sentinelInPtr *ptrsIn, sentinelOutPtr *ptrsOut, char *data, char *dataEnd, intptr_t offset) {
	char *ptr, *next;
	if (ptrsIn) {
		ptr = data;
		for (sentinelInPtr *p = ptrsIn; p->field; p++) {
			char **field = (char **)p->field;
			int size = p->size != -1 ? p->size : (p->size = *field ? (int)strlen(*field) + 1 : 0);
			next = ptr + size;
			if (size && next <= dataEnd) {
				memcpy(ptr, *field, size);
				*field = ptr + offset;
				ptr = next;
			}
			else return nullptr;
		}
	}
	if (ptrsOut) {
		ptr = ptrsOut[0].field != (char *)-1 ? data : ptr;
		for (sentinelOutPtr *p = ptrsOut; p->field; p++) {
			char **field = (char **)p->field;
			int size = p->size != -1 ? p->size : dataEnd - ptr;
			next = data + size;
			if (next <= dataEnd) {
				*field = data + offset;
				ptr = next;
			}
			else return nullptr;
		}
	}
	return data;
}

static sentinelMap *_sentinelClientMap = nullptr;
static intptr_t _sentinelClientMapOffset = 0;
void sentinelClientSend(sentinelMessage *msg, int msgLength, sentinelInPtr *ptrsIn = nullptr, sentinelOutPtr *ptrsOut = nullptr) {
#ifndef _WIN64
	printf("Sentinel client currently only works in x64.\n"); abort();
#else
	sentinelMap *map = _sentinelClientMap;
	if (!map) {
		printf("sentinel: client map not defined. did you start sentinel?\n"); exit(0);
	}

	// ATTACH
#if __OS_WIN
	long id = InterlockedAdd((long *)&map->setId, SENTINEL_MSGSIZE) - SENTINEL_MSGSIZE;
#elif __OS_UNIX
	long id = __sync_fetch_and_add((long *)&map->SetId, SENTINEL_MSGSIZE) - SENTINEL_MSGSIZE;
#endif
	sentinelCommand *cmd = (sentinelCommand *)&map->data[id % sizeof(map->data)];
	if (cmd->magic != SENTINEL_MAGIC) {
		printf("bad sentinel magic"); exit(1);
	}
	int *unknown = &cmd->unknown; volatile long *control = (volatile long *)&cmd->control;
	HOST_SPINLOCK(SENTINELCONTROL_DEVICE, SENTINELCONTROL_NORMAL);

	// PREPARE
	cmd->length = msgLength;
	char *data = cmd->data + ROUND8_(msgLength), *dataEnd = data + msg->size;
	if (((ptrsIn || ptrsOut) && !(data = preparePtrs(ptrsIn, ptrsOut, data, dataEnd, _sentinelClientMapOffset))) ||
		(msg->prepare && !msg->prepare(msg, data, dataEnd, _sentinelClientMapOffset))) {
		printf("msg too long"); exit(0);
	}
	memcpy(cmd->data, msg, msgLength);
	//printf("msg: %d[%d]'", msg->op, msgLength); for (int i = 0; i < msgLength; i++) printf("%02x", ((char *)msg)[i] & 0xff); printf("'\n");
	*unknown = 0; *control = SENTINELCONTROL_DEVICERDY;

	// FLOW-WAIT
	if (msg->flow & SENTINELFLOW_WAIT) {
		HOST_SPINLOCK(SENTINELCONTROL_DEVICE2, SENTINELCONTROL_HOSTRDY);
		memcpy(msg, cmd->data, msgLength);
	}
	*control = SENTINELCONTROL_NORMAL;
#endif
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

static __forceinline__ int getprocessid_() { host_getprocessid msg; return msg.RC; }

static char *sentinelClientRedirPipelineArgs[] = { (char *)"^0" };
void sentinelClientRedir(pipelineRedir *redir) {
#if __OS_WIN
	HANDLE process = OpenProcess(PROCESS_DUP_HANDLE, FALSE, getprocessid_());
	CreatePipeline(1, sentinelClientRedirPipelineArgs, nullptr, &redir[1].Input, &redir[1].Output, &redir[1].Error, process, redir);
#elif __OS_UNIX
#endif
}

#endif