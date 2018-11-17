#include <stdiocu.h>
#include <stringcu.h>
#include <sentinel.h>
#include <assert.h>

enum {
	MODULE_SIMPLE = 500,
	MODULE_STRING,
	MODULE_CUSTOM,
	MODULE_JUMBOIN,
	MODULE_JUMBOOUT,
};

struct module_simple {
	sentinelMessage Base;
	int Value;
	__device__ module_simple(bool wait, int value) : Base(MODULE_SIMPLE, wait ? FLOW_WAIT : FLOW_NONE), Value(value) { sentinelDeviceSend(&Base, sizeof(module_simple)); }
	int RC;
};

struct module_string {
	sentinelMessage Base;
	const char *Str;
	__device__ module_string(bool wait, const char *str) : Base(MODULE_STRING, wait ? FLOW_WAIT : FLOW_NONE, SENTINEL_CHUNK), Str(str) { sentinelDeviceSend(&Base, sizeof(module_string), PtrsIn); }
	int RC;
	sentinelInPtr PtrsIn[2] = {
		{ &Str, -1 },
		nullptr
	};
};

struct module_custom {
	static __forceinline__ __device__ char *Prepare(module_custom *t, char *data, char *dataEnd, intptr_t offset) {
		int strLength = t->Str ? (int)strlen(t->Str) + 1 : 0;
		char *str = data;
		char *end = data += strLength;
		if (end > dataEnd) return nullptr;
		memcpy(str, t->Str, strLength);
		t->Str = str + offset;
		return end;
	}
	sentinelMessage Base;
	const char *Str;
	__device__ module_custom(bool wait, const char *str) : Base(MODULE_CUSTOM, wait ? FLOW_WAIT : FLOW_NONE, SENTINEL_CHUNK, SENTINELPREPARE(Prepare)), Str(str) { sentinelDeviceSend(&Base, sizeof(module_custom)); }
	int RC;
};

struct module_jumboin {
	/*static __forceinline__ __device__ bool Postfix(module_jumboin *t, intptr_t offset) {
		char *ptr = (char *)t->Ptr - offset;
		if (t->RC > 0) memcpy(t->Buf, ptr, t->Size * t->RC);
		return true;
	}*/
	sentinelMessage Base;
	const char *Str;
	__device__ module_jumboin(bool wait, const char *str, char *ptr, size_t size) : Base(MODULE_JUMBOIN, wait ? FLOW_WAIT : FLOW_NONE, SENTINEL_CHUNK) { PtrsIn[1].size = size; sentinelDeviceSend(&Base, sizeof(module_jumboin), PtrsIn); }
	int RC;
	void *Ptr;
	sentinelInPtr PtrsIn[3] = {
		{ &Str, -1 },
		{ &Ptr, 0 },
		nullptr
	};
};

struct module_jumboout {
	static __forceinline__ __device__ char *Prepare(module_jumboout *t, char *data, char *dataEnd, intptr_t offset) {
		size_t size = t->Size;
		char *ptr = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += size);
		if (end > dataEnd) return nullptr;
		memcpy(ptr, t->Ptr, size);
		t->Ptr = ptr + offset;
		return end;
	}
	sentinelMessage Base;
	void *Ptr; size_t Size;
	__device__ module_jumboout(char *ptr, size_t size) : Base(MODULE_JUMBOOUT, FLOW_WAIT, SENTINEL_CHUNK, SENTINELPREPARE(Prepare)) { sentinelDeviceSend(&Base, sizeof(module_jumboout)); }
	int RC;
};

bool sentinelModuleExecutor(void *tag, sentinelMessage *data, int length, char *(**hostPrepare)(void*, char*, char*, intptr_t)) {
	switch (data->OP) {
	case MODULE_SIMPLE: { module_simple *msg = (module_simple *)data; msg->RC = msg->Value; return true; }
	case MODULE_STRING: { module_string *msg = (module_string *)data; msg->RC = (int)strlen(msg->Str); return true; }
	case MODULE_CUSTOM: { module_custom *msg = (module_custom *)data; msg->RC = (int)strlen(msg->Str); return true; }
	case MODULE_JUMBOIN: { module_jumboin *msg = (module_jumboin *)data; msg->RC = (int)strlen(msg->Str); return true; }
	case MODULE_JUMBOOUT: { module_jumboout *msg = (module_jumboout *)data; msg->RC = 0; return true; }
	}
	return false;
}
static sentinelExecutor _moduleExecutor = { nullptr, "module", sentinelModuleExecutor, nullptr };

static __global__ void g_sentinel_test1() {
	printf("sentinel_test1\n");

	//// SENTINELDEVICESEND ////
	//	extern __device__ void sentinelDeviceSend(sentinelMessage *msg, int msgLength);
	module_simple a0(true, 1); int a0a = a0.RC; assert(a0a == 1);
	module_string a1(true, "test"); int a1a = a1.RC; assert(a1a == 4);
	module_custom a2(true, "test"); int a2a = a2.RC; assert(a2a == 4);

	// JUMBO
	char jumbosmall[2048], jumbolarge[9046]; memset(jumbosmall, 1, sizeof(jumbosmall)); memset(jumbolarge, 2, sizeof(jumbolarge));
	module_jumboin b0(true, "test", jumbosmall, sizeof(jumbosmall));
	int b0a = b0.RC; assert(b0a == 4);
	//module_jumboin b1(true, "test", jumbolarge, sizeof(jumbolarge));
	//int b1a = b1.RC; //assert(b1a == 4);
}

cudaError_t sentinel_test1() {
	sentinelRegisterExecutor(&_moduleExecutor);
	g_sentinel_test1<<<1, 1>>>(); return cudaDeviceSynchronize();
}

//// SENTINELDEFAULTEXECUTOR ////
//	extern bool sentinelDefaultExecutor(void *tag, sentinelMessage *data, int length, char *(**hostPrepare)(void*,char*,char*,intptr_t));

//// SENTINELSERVERINITIALIZE, SENTINELSERVERSHUTDOWN ////
//	extern void sentinelServerInitialize(sentinelExecutor *executor = nullptr, char *mapHostName = SENTINEL_NAME, bool hostSentinel = true, bool deviceSentinel = true);
//	extern void sentinelServerShutdown();

//// SENTINELDEVICESEND ////
//	extern __device__ void sentinelDeviceSend(sentinelMessage *msg, int msgLength);

//// SENTINELCLIENTINITIALIZE, SENTINELCLIENTSHUTDOWN ////
//	extern void sentinelClientInitialize(char *mapHostName = SENTINEL_NAME);
//	extern void sentinelClientShutdown();

//// SENTINELCLIENTSEND ////
//	extern void sentinelClientSend(sentinelMessage *msg, int msgLength);

//// SENTINELFINDEXECUTOR, SENTINELREGISTEREXECUTOR, SENTINELUNREGISTEREXECUTOR ////
//	extern sentinelExecutor *sentinelFindExecutor(const char *name, bool forDevice = true);
//	extern void sentinelRegisterExecutor(sentinelExecutor *exec, bool makeDefault = false, bool forDevice = true);
//	extern void sentinelUnregisterExecutor(sentinelExecutor *exec, bool forDevice = true);

//// SENTINELREGISTERFILEUTILS ////
//	extern void sentinelRegisterFileUtils();
