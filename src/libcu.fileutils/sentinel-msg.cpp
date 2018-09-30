#define _CRT_SECURE_NO_WARNINGS
#include "sentinel-fileutilsmsg.h"

int dcat(pipelineRedir redir, char *str);
int dchgrp(pipelineRedir redir, char *str, int gid);
struct group *dchgrp_getgrnam(pipelineRedir redir, char *str);
int dchmod(pipelineRedir redir, char *str, int mode);
int dchown(pipelineRedir redir, char *str, int uid);
struct passwd *dchown_getpwnam_(pipelineRedir redir, char *str);
int dcmp(pipelineRedir redir, char *str, char *str2);
int dcp(pipelineRedir redir, char *str, char *str2, bool setModes);
bool dcp_isadir_(pipelineRedir redir, char *str);
int dgrep(pipelineRedir redir, char *str, char *str2, bool ignoreCase, bool tellName, bool tellLine);
int dls(pipelineRedir redir, char *str, int flags, bool endSlash);
int dmkdir(pipelineRedir redir, char *str, unsigned short mode);
int dmore(pipelineRedir redir, char *str, int fd);
int dmv(pipelineRedir redir, char *str, char *str2);
int drm(pipelineRedir redir, char *str);
int drmdir(pipelineRedir redir, char *str);
int dpwd(pipelineRedir redir, char *str);
int dcd(pipelineRedir redir, char *str);

extern "C" bool sentinelFileUtilsExecutor(void *tag, sentinelMessage *data, int length, char *(**hostPrepare)(void*, char*, char*, intptr_t)) {
	if (data->OP < FILEUTILS_DCAT || data->OP > FILEUTILS_DCD) return false;
	switch (data->OP) {
	case FILEUTILS_DCAT: { fileutils_dcat *msg = (fileutils_dcat *)data; msg->RC = dcat(msg->Base.Redir, msg->Str); return true; }
	case FILEUTILS_DCHGRP: { fileutils_dchgrp *msg = (fileutils_dchgrp *)data; msg->RC = dchgrp(msg->Base.Redir, msg->Str, msg->Gid); return true; }
	case FILEUTILS_GETGRNAM: { fileutils_getgrnam *msg = (fileutils_getgrnam *)data; msg->RC = dchgrp_getgrnam(msg->Base.Redir, msg->Str); return true; }
	case FILEUTILS_DCHMOD: { fileutils_dchmod *msg = (fileutils_dchmod *)data; msg->RC = dchmod(msg->Base.Redir, msg->Str, msg->Mode); return true; }
	case FILEUTILS_DCHOWN: { fileutils_dchown *msg = (fileutils_dchown *)data; msg->RC = dchown(msg->Base.Redir, msg->Str, msg->Uid); return true; }
	case FILEUTILS_GETPWNAM: { fileutils_getpwnam *msg = (fileutils_getpwnam *)data; msg->RC = dchown_getpwnam_(msg->Base.Redir, msg->Str); return true; }
	case FILEUTILS_DCMP: { fileutils_dcmp *msg = (fileutils_dcmp *)data; msg->RC = dcmp(msg->Base.Redir, msg->Str, msg->Str2); return true; }
	case FILEUTILS_DCP: { fileutils_dcp *msg = (fileutils_dcp *)data; msg->RC = dcp(msg->Base.Redir, msg->Str, msg->Str2, msg->SetModes); return true; }
	case FILEUTILS_ISADIR: { fileutils_isadir *msg = (fileutils_isadir *)data; msg->RC = dcp_isadir_(msg->Base.Redir, msg->Str); return true; }
	case FILEUTILS_DGREP: { fileutils_dgrep *msg = (fileutils_dgrep *)data; msg->RC = dgrep(msg->Base.Redir, msg->Str, msg->Str2, msg->IgnoreCase, msg->TellName, msg->TellLine); return true; }
	case FILEUTILS_DLS: { fileutils_dls *msg = (fileutils_dls *)data; msg->RC = dls(msg->Base.Redir, msg->Str, msg->Flags, msg->EndSlash); return true; }
	case FILEUTILS_DMKDIR: { fileutils_dmkdir *msg = (fileutils_dmkdir *)data; msg->RC = dmkdir(msg->Base.Redir, msg->Str, msg->Mode); return true; }
	case FILEUTILS_DMORE: { fileutils_dmore *msg = (fileutils_dmore *)data; msg->RC = dmore(msg->Base.Redir, msg->Str, msg->Fd); return true; }
	case FILEUTILS_DMV: { fileutils_dmv *msg = (fileutils_dmv *)data; msg->RC = dmv(msg->Base.Redir, msg->Str, msg->Str2); return true; }
	case FILEUTILS_DRM: { fileutils_drm *msg = (fileutils_drm *)data; msg->RC = drm(msg->Base.Redir, msg->Str); return true; }
	case FILEUTILS_DRMDIR: { fileutils_drmdir *msg = (fileutils_drmdir *)data; msg->RC = drmdir(msg->Base.Redir, msg->Str); return true; }
	case FILEUTILS_DPWD: { fileutils_dpwd *msg = (fileutils_dpwd *)data; msg->RC = dpwd(msg->Base.Redir, msg->Ptr); return true; }
	case FILEUTILS_DCD: { fileutils_dcd *msg = (fileutils_dcd *)data; msg->RC = dcd(msg->Base.Redir, msg->Str); return true; }
	}
	return false;
}