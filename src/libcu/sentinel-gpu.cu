#include <crtdefscu.h>
#include <stringcu.h>
#include <sentinel.h>
#include <ext/mutex.h>

__BEGIN_DECLS;

#if HAS_DEVICESENTINEL

static __device__ void executeTrans(sentinelCommand *cmd, int size, sentinelInPtr *listIn, sentinelOutPtr *listOut, intptr_t offset, char *&trans);

static __device__ char *preparePtrs(sentinelInPtr *ptrsIn, sentinelOutPtr *ptrsOut, sentinelCommand *cmd, char *data, char *dataEnd, intptr_t offset, sentinelOutPtr *&listOut, char *&trans) {
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
		// if field == -1, append from previous ptr
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
	if (ptrsIn)
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
	if (ptrsOut)
		for (sentinelOutPtr *p = ptrsOut; p->field; p++) {
			// if field == -1, continue
			if (p->field == (char *)-1)
				continue;
			char **buf = (char **)p->buf;
			if (!*buf)
				continue;
			char **field = (char **)p->field;
			char *ptr = *field - offset;
			int *sizeField = (int *)p->sizeField;
			int size = !sizeField ? p->size : *sizeField;
			if (size > 0) memcpy(*buf, ptr, size);
		}
	return true;
}

__device__ volatile unsigned int _sentinelMapId;
__constant__ const sentinelMap *_sentinelDeviceMap[SENTINEL_DEVICEMAPS];
__device__ void sentinelDeviceSend(sentinelMessage *msg, int msgLength, sentinelInPtr *ptrsIn, sentinelOutPtr *ptrsOut) {
	const sentinelMap *map = _sentinelDeviceMap[_sentinelMapId++ % SENTINEL_DEVICEMAPS];
	if (!map)
		panic("sentinel: device map not defined. did you start sentinel?\n");

	// ATTACH
	long id = atomicAdd((int *)&map->setId, SENTINEL_MSGSIZE);
	sentinelCommand *cmd = (sentinelCommand *)&map->data[id % sizeof(map->data)];
	if (cmd->magic != SENTINEL_MAGIC)
		panic("bad sentinel magic");
	volatile long *control = &cmd->control; intptr_t offset = map->offset; char *trans = nullptr;
	mutexSpinLock(nullptr, control, SENTINELCONTROL_NORMAL, SENTINELCONTROL_DEVICE);

	// PREPARE
	char *data = cmd->data + ROUND8_(msgLength), *dataEnd = data + msg->size;
	sentinelOutPtr *listOut = nullptr;
	if (((ptrsIn || ptrsOut) && !(data = preparePtrs(ptrsIn, ptrsOut, cmd, data, dataEnd, offset, listOut, trans))) ||
		(msg->prepare && !msg->prepare(msg, data, dataEnd, offset)))
		panic("msg too long");
	cmd->length = msgLength; memcpy(cmd->data, msg, msgLength);
	//printf("msg: %d[%d]'", msg->op, msgLength); for (int i = 0; i < msgLength; i++) printf("%02x", ((char *)msg)[i] & 0xff); printf("'\n");
	mutexSet(control, SENTINELCONTROL_DEVICERDY);

	// FLOW-WAIT
	if (msg->flow & SENTINELFLOW_WAIT) {
		mutexSpinLock(nullptr, control, SENTINELCONTROL_HOSTRDY, SENTINELCONTROL_DEVICEWAIT);
		//if (listOut) executeTrans(cmd, 0, nullptr, listOut, offset, trans);
		cmd->length = msgLength; memcpy(msg, cmd->data, msgLength);
		if ((ptrsOut && !postfixPtrs(ptrsOut, cmd, offset)) ||
			(msg->postfix && !msg->postfix(msg, offset)))
			panic("postfix error");
		mutexSet(control, SENTINELCONTROL_NORMAL);
	}
}

static __device__ void executeTrans(sentinelCommand *cmd, int size, sentinelInPtr *listIn, sentinelOutPtr *listOut, intptr_t offset, char *&trans) {
	volatile long *control = &cmd->control;
	char *data = cmd->data, *ptr = trans;
	if (size) {
		*(int *)data = size;
		mutexSet(control, SENTINELCONTROL_TRANSSIZE);
		mutexSpinLock(nullptr, control, SENTINELCONTROL_TRANRDY, SENTINELCONTROL_TRANDONE);
		ptr = trans = *(char **)data;
	}
	if (listIn)
		for (sentinelInPtr *p = listIn; p; p = (sentinelInPtr *)p->unknown) {
			char **field = (char **)p->field;
			const char *v = (const char *)*field; int size = p->size, remain = size, length = 0;
			while (remain > 0) {
				length = cmd->length = remain > SENTINEL_MSGSIZE ? SENTINEL_MSGSIZE : remain;
				memcpy(data, (void *)v, length); remain -= length; v += length;
				mutexSet(control, SENTINELCONTROL_TRANSIN);
				mutexSpinLock(nullptr, control, SENTINELCONTROL_TRANRDY, SENTINELCONTROL_TRANDONE);
			}
			*field = ptr; ptr += size;
			p->unknown = nullptr;
		}
	if (listOut)
		for (sentinelOutPtr *p = listOut; p; p = (sentinelOutPtr *)p->unknown) {
			char **field = (char **)p->field;
			const char *v = (const char *)*field; int size = p->size, remain = size, length = 0;
			while (remain > 0) {
				length = cmd->length = remain > SENTINEL_MSGSIZE ? SENTINEL_MSGSIZE : remain;
				memcpy((void *)v, data, length); remain -= length; v += length;
				mutexSet(control, SENTINELCONTROL_TRANSOUT);
				mutexSpinLock(nullptr, control, SENTINELCONTROL_TRANRDY, SENTINELCONTROL_TRANDONE);
			}
			*field = ptr; ptr += size;
			p->unknown = nullptr;
		}
}

#endif

__END_DECLS;
