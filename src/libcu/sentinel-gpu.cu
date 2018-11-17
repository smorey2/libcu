#include <crtdefscu.h>
#include <stdiocu.h>
#include <stdlibcu.h>
#include <sentinel.h>

__BEGIN_DECLS;

#if HAS_DEVICESENTINEL

//static __forceinline__ __device__ char *Prepare(fcntl_open *t, char *data, char *dataEnd, intptr_t offset) {
//	int strLength = t->Str ? (int)strlen(t->Str) + 1 : 0;
//	char *str = (char *)(data += ROUND8_(sizeof(*t)));
//	char *end = (char *)(data += strLength);
//	if (end > dataEnd) return nullptr;
//	memcpy(str, t->Str, strLength);
//	if (t->Str) t->Str = str + offset;
//	return end;
//}

//static __forceinline__ __device__ char *Prepare(fcntl_fstat *t, char *data, char *dataEnd, intptr_t offset) {
//	char *ptr = (char *)(data += ROUND8_(sizeof(*t)));
//	char *end = (char *)(data += SENTINEL_CHUNK);
//	if (end > dataEnd) return nullptr;
//	t->Ptr = ptr + offset;
//	return end;
//}

static __device__ char *preparePtrs(sentinelInPtr *ptrsIn, sentinelOutPtr *ptrsOut, char *data, char *dataEnd, intptr_t offset) {
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

//static __forceinline__ __device__ bool Postfix(fcntl_fstat *t, intptr_t offset) {
//	char *ptr = (char *)t->Ptr - offset;
//	if (!t->Bit64) memcpy(t->Buf, ptr, sizeof(struct stat));
//	else memcpy(t->Buf64, ptr, sizeof(struct _stat64));
//	return true;
//}
//static __forceinline__ __device__ bool Postfix(unistd_read *t, intptr_t offset) {
//	char *ptr = (char *)t->Ptr - offset;
//	if (t->RC > 0) memcpy(t->Buf, ptr, t->RC);
//	return true;
//}

static __device__ bool postfixPtrs(sentinelOutPtr *ptrsOut, intptr_t offset) {
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

__device__ volatile unsigned int _sentinelMapId;
__constant__ const sentinelMap *_sentinelDeviceMap[SENTINEL_DEVICEMAPS];
__device__ void sentinelDeviceSend(sentinelMessage *msg, int msgLength, sentinelInPtr *ptrsIn, sentinelOutPtr *ptrsOut) {
	unsigned int s_;
	const sentinelMap *map = _sentinelDeviceMap[_sentinelMapId++ % SENTINEL_DEVICEMAPS];
	if (!map)
		panic("sentinel: device map not defined. did you start sentinel?\n");

	// ATTACH
	long id = atomicAdd((int *)&map->SetId, SENTINEL_MSGSIZE);
	sentinelCommand *cmd = (sentinelCommand *)&map->Data[id % sizeof(map->Data)]; //cmd->Data = (char *)cmd + ROUND8_(sizeof(sentinelCommand));
	if (cmd->Magic != SENTINEL_MAGIC)
		panic("Bad Sentinel Magic");
	volatile long *control = (volatile long *)&cmd->Control;
	/* spin-lock */ do { s_ = *control; /*printf("%d ", s_);*/ __syncthreads(); } while (s_ != 0); __syncthreads();
	//*control = 1; // device in-progress

	// PREPARE
	cmd->Length = msgLength;
	char *data = cmd->Data + ROUND8_(msgLength), *dataEnd = data + msg->Size;
	if ((ptrsIn || ptrsOut) && !(data = preparePtrs(ptrsIn, ptrsOut, data, dataEnd, map->Offset)))
		panic("msg too long");
	if (msg->Prepare && !msg->Prepare(msg, data, dataEnd, map->Offset))
		panic("msg too long");

	// FLOW-OUT
	//if (msg->Flow & FLOW_JUMBOOUT) {
	//	while (true) {
	//		/* spin-lock */ do { s_ = *control; /*printf("%d ", s_);*/ __syncthreads(); } while (s_ != 10); __syncthreads();
	//	}
	//}
	memcpy(cmd->Data, msg, msgLength);
	//printf("Msg: %d[%d]'", msg->OP, msgLength); for (int i = 0; i < msgLength; i++) printf("%02x", ((char *)msg)[i] & 0xff); printf("'\n");
	*control = 2; // client signal that data is ready to process

	// FLOW-IN
	//if (msg->Flow & FLOW_JUMBOIN) {
	//	while (true) {
	//		/* spin-lock */ do { s_ = *control; /*printf("%d ", s_);*/ __syncthreads(); } while (s_ != 10); __syncthreads();
	//	}
	//}

	// FLOW-WAIT
	if (msg->Flow & FLOW_WAIT) {
		/* spin-lock */ do { s_ = *control; /*printf("%d ", s_);*/ __syncthreads(); } while (s_ != 4); __syncthreads();
		memcpy(msg, cmd->Data, msgLength);
		if (ptrsOut && !postfixPtrs(ptrsOut, map->Offset))
			panic("postfix error");
		if (msg->Postfix && !msg->Postfix(msg, map->Offset))
			panic("postfix error");
	}
	*control = 0; // normal state
}

#endif

__END_DECLS;
