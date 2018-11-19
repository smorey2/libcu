#include <crtdefscu.h>
#include <stdiocu.h>
#include <stdlibcu.h>
#include <sentinel.h>

#define DEVICE_SPINLOCK(SET, WHEN) do { s_ = *control; /*printf("%d ", s_);*/ __syncthreads(); } while (s_ != WHEN); *control = SET; __syncthreads();

__BEGIN_DECLS;

#if HAS_DEVICESENTINEL

static __device__ void executeTrans(sentinelCommand *cmd, int size, sentinelInPtr *listIn, sentinelOutPtr *listOut, intptr_t offset);
static __device__ char *preparePtrs(sentinelInPtr *ptrsIn, sentinelOutPtr *ptrsOut, sentinelCommand *cmd, char *data, char *dataEnd, intptr_t offset, sentinelOutPtr *&transListOut) {
	char *ptr = data, *next;
	int transSize = 0;
	sentinelInPtr *listIn = nullptr;
	sentinelOutPtr *listOut = nullptr;

	// PREPARE
	if (ptrsIn)
		for (sentinelInPtr *p = ptrsIn; p->field; p++) {
			char **field = (char **)p->field;
			int size = p->size != -1 ? p->size : (p->size = *field ? (int)strlen(*field) + 1 : 0);
			next = ptr + size;
			if (!size)
				p->unknown = nullptr;
			else if (next <= dataEnd) {
				p->unknown = ptr;
				ptr = next;
			}
			else {
				p->unknown = listIn; listIn = p;
				transSize += size;
			}
		}
	if (ptrsOut) {
		ptr = ptrsOut[0].field == (char *)-1 ? ptr : data;
		for (sentinelOutPtr *p = ptrsOut; p->field; p++) {
			char **field = (char **)p->field;
			int size = p->size != -1 ? p->size : (int)(dataEnd - ptr);
			next = ptr + size;
			if (!size) {}
			else if (next <= dataEnd) {
				*field = ptr + offset;
				ptr = next;
			}
			else {
				p->unknown = listOut; listOut = p;
				transSize += size;
			}
		}
		transListOut = listOut;
	}

	// TRANSFER IN
	if (transSize)
		executeTrans(cmd, transSize, listIn, nullptr, offset);

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

static __device__ bool postfixPtrs(sentinelOutPtr *ptrsOut, sentinelCommand *cmd, intptr_t offset) {
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

__device__ volatile unsigned int _sentinelMapId;
__constant__ const sentinelMap *_sentinelDeviceMap[SENTINEL_DEVICEMAPS];
__device__ void sentinelDeviceSend(sentinelMessage *msg, int msgLength, sentinelInPtr *ptrsIn, sentinelOutPtr *ptrsOut) {
	unsigned int s_;
	const sentinelMap *map = _sentinelDeviceMap[_sentinelMapId++ % SENTINEL_DEVICEMAPS];
	if (!map)
		panic("sentinel: device map not defined. did you start sentinel?\n");

	// ATTACH
	long id = atomicAdd((int *)&map->setId, SENTINEL_MSGSIZE);
	sentinelCommand *cmd = (sentinelCommand *)&map->data[id % sizeof(map->data)];
	if (cmd->magic != SENTINEL_MAGIC)
		panic("bad sentinel magic");
	int *unknown = &cmd->unknown; volatile long *control = (volatile long *)&cmd->control; intptr_t offset = map->offset;
	DEVICE_SPINLOCK(SENTINELCONTROL_DEVICE, SENTINELCONTROL_NORMAL);

	// PREPARE
	cmd->length = msgLength;
	char *data = cmd->data + ROUND8_(msgLength), *dataEnd = data + msg->size;
	sentinelOutPtr *transListOut = nullptr;
	if (((ptrsIn || ptrsOut) && !(data = preparePtrs(ptrsIn, ptrsOut, cmd, data, dataEnd, offset, transListOut))) ||
		(msg->prepare && !msg->prepare(msg, data, dataEnd, offset)))
		panic("msg too long");
	cmd->length = msgLength;
	memcpy(cmd->data, msg, msgLength);
	//printf("msg: %d[%d]'", msg->op, msgLength); for (int i = 0; i < msgLength; i++) printf("%02x", ((char *)msg)[i] & 0xff); printf("'\n");
	*unknown = 0; *control = SENTINELCONTROL_DEVICERDY;

	// FLOW-WAIT
	if (msg->flow & SENTINELFLOW_WAIT) {
		DEVICE_SPINLOCK(SENTINELCONTROL_DEVICE, SENTINELCONTROL_HOSTRDY);
		executeTrans(cmd, 0, nullptr, transListOut, offset);
		cmd->length = msgLength;
		memcpy(msg, cmd->data, msgLength);
		if ((ptrsOut && !postfixPtrs(ptrsOut, cmd, offset)) ||
			(msg->postfix && !msg->postfix(msg, offset)))
			panic("postfix error");
		*unknown = 0; *control = SENTINELCONTROL_DEVICERDY;
	}
	*control = SENTINELCONTROL_NORMAL;
}

static __device__ void executeTrans(sentinelCommand *cmd, int size, sentinelInPtr *listIn, sentinelOutPtr *listOut, intptr_t offset) {
	unsigned int s_;
	int *unknown = &cmd->unknown; volatile long *control = (volatile long *)&cmd->control;
	char *data = cmd->data;
	// create memory
	if (size) {
		*(int *)data = size;
		*unknown = 1; *control = SENTINELCONTROL_DEVICERDY;
		DEVICE_SPINLOCK(SENTINELCONTROL_DEVICE, SENTINELCONTROL_HOSTRDY);
	}
	char *ptr = *(char **)data;
	// transfer
	if (listIn) {
		for (sentinelInPtr *p = listIn; p; p = (sentinelInPtr *)p->unknown) {
			char **field = (char **)p->field;
			int size = p->size, length = 0; const char *v = (const char *)*field;
			while (size > 0) {
				length = cmd->length = size > SENTINEL_MSGSIZE ? SENTINEL_MSGSIZE : size;
				memcpy(data, (void *)v, length); size -= length; v += length;
				*unknown = 2; *control = SENTINELCONTROL_DEVICERDY;
				DEVICE_SPINLOCK(SENTINELCONTROL_DEVICE, SENTINELCONTROL_HOSTRDY);
			}
			*field = ptr; ptr += size;
			p->unknown = nullptr;
		}
		*unknown = 0; *control = SENTINELCONTROL_DEVICERDY;
	}
	if (listOut) {
		for (sentinelOutPtr *p = listOut; p; p = (sentinelOutPtr *)p->unknown) {
			char **field = (char **)p->field;
			int size = p->size, length = 0; const char *v = (const char *)*field;
			while (size > 0) {
				length = cmd->length = size > SENTINEL_MSGSIZE ? SENTINEL_MSGSIZE : size;
				memcpy((void *)v, data, length); size -= length; v += length;
				*unknown = 3; *control = SENTINELCONTROL_DEVICERDY;
				DEVICE_SPINLOCK(SENTINELCONTROL_DEVICE, SENTINELCONTROL_HOSTRDY);
			}
			*field = ptr; ptr += size;
			p->unknown = nullptr;
		}
	}
}

#endif

__END_DECLS;
