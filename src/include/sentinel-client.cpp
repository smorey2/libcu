#include <sentinel.h>
#include <sentinel-hostmsg.h>
#if __OS_WIN
#include <windows.h>
#elif __OS_UNIX
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#endif
#include <stdio.h>

#if HAS_HOSTSENTINEL

static sentinelMap *_sentinelClientMap = nullptr;
static intptr_t _sentinelClientMapOffset = 0;
void sentinelClientSend(sentinelMessage *msg, int msgLength) {
#ifndef _WIN64
	printf("Sentinel currently only works in x64.\n");
	abort();
#else
	sentinelMap *map = _sentinelClientMap;
	if (!map) {
		printf("sentinel: client map not defined. did you start sentinel?\n");
		exit(0);
	}

	// ATTACH
#if __OS_WIN
	long id = InterlockedAdd((long *)&map->SetId, SENTINEL_MSGSIZE) - SENTINEL_MSGSIZE;
#elif __OS_UNIX
	long id = __sync_fetch_and_add((long *)&map->SetId, SENTINEL_MSGSIZE) - SENTINEL_MSGSIZE;
#endif
	sentinelCommand *cmd = (sentinelCommand *)&map->Data[id % sizeof(map->Data)]; //cmd->Data = (char *)cmd + ROUND8_(sizeof(sentinelCommand));
	if (cmd->Magic != SENTINEL_MAGIC) {
		printf("Bad Sentinel Magic");
		exit(1);
	}
	volatile long *control = (volatile long *)&cmd->Control;
#if __OS_WIN
	/* spin-lock */ while (InterlockedCompareExchange((long *)control, 1, 0) != 0) {} // device in-progress
#elif __OS_UNIX
	/* spin-lock */ while (__sync_val_compare_and_swap((long *)control, 1, 0) != 0) {} // device in-progress
#endif


	// PREPARE
	cmd->Length = msgLength;
	if (msg->Prepare && !msg->Prepare(msg, cmd->Data, cmd->Data + ROUND8_(msgLength) + msg->Size, _sentinelClientMapOffset)) {
		printf("msg too long");
		exit(0);
	}

	// FLOW-OUT
	if (msg->Flow & FLOW_JUMBOOUT) {
		while (true) {
#if __OS_WIN
			/* spin-lock */ while (InterlockedCompareExchange((long *)control, 9, 10) != 10) {}
#elif __OS_UNIX
			/* spin-lock */ while (__sync_val_compare_and_swap((long *)control, 9, 10) != 10) {}
#endif
		}
	}
	memcpy(cmd->Data, msg, msgLength);
	//printf("Msg: %d[%d]'", msg->OP, msgLength); for (int i = 0; i < msgLength; i++) printf("%02x", ((char *)msg)[i] & 0xff); printf("'\n");
	*control = 2; // client signal that data is ready to process

	// FLOW-IN
	if (msg->Flow & FLOW_JUMBOIN) {
		while (true) {
#if __OS_WIN
			/* spin-lock */ while (InterlockedCompareExchange((long *)control, 9, 10) != 10) {}
#elif __OS_UNIX
			/* spin-lock */ while (__sync_val_compare_and_swap((long *)control, 9, 10) != 10) {}
#endif
		}
	}

	// FLOW-WAIT
	if (msg->Flow & FLOW_WAIT) {
#if __OS_WIN
		/* spin-lock */ while (InterlockedCompareExchange((long *)control, 9, 4) != 4) {}
#elif __OS_UNIX
		/* spin-lock */ while (__sync_val_compare_and_swap((long *)control, 9, 4) != 4) {}
#endif
		memcpy(msg, cmd->Data, msgLength);
	}
	*control = 0; // normal state
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
		printf("(%d) Could not connect to Sentinel host. Please ensure host application is running.\n", GetLastError());
		exit(1);
	}
	_clientMap = (int *)MapViewOfFile(_clientMapHandle, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(sentinelMap) + MEMORY_ALIGNMENT);
	if (!_clientMap) {
		printf("(%d) Could not map view of file.\n", GetLastError());
		CloseHandle(_clientMapHandle);
		exit(1);
	}
	_sentinelClientMap = (sentinelMap *)ROUNDN_(_clientMap, MEMORY_ALIGNMENT);
	_sentinelClientMapOffset = (intptr_t)((char *)_sentinelClientMap->Offset - (char *)_sentinelClientMap);
#elif __OS_UNIX
	struct stat sb;
	int fd = open(mapHostName, O_RDONLY);
	if (fd == -1) { perror("open"); exit(1); }
	if (fstat(fd, &sb) == -1) { perror("fstat"); exit(1); }
	if (!S_ISREG(sb.st_mode)) { fprintf(stderr, "%s is not a file\n", mapHostName); exit(1); }
	_clientMap = mmap(NULL, sizeof(sentinelMap) + MEMORY_ALIGNMENT, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, fd, 0);
	if (!_clientMap) {
		printf("Could not connect to Sentinel host. Please ensure host application is running.\n");
		exit(1);
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