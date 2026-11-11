#include"Graph.h"
#include"Help.h"
#include<vector>
#include <windows.h>


void Barf(const char* s)
{
    fprintf(stderr, "%s\n", s);
    exit(-1);
}

FILE* OpenFile(const char* c)
{
    FILE* f;

    if (fopen_s(&f, c, "r") == 0 && f != nullptr) {
        return f;
    }

    fprintf(stderr, "Cannot open input file: %s\n", c == nullptr ? "(null)" : c);
    return NULL;
}



char ReadChar(FILE* f) {
    char c;

    do {
        c = getc(f);
    } while (isspace(c));
    return c;
}


int EOF_Test(FILE* f) {
    char c;

    c = ReadChar(f);
    if (c == EOF)
        return TRUE;
    ungetc(c, f);
    return FALSE;
}

int GetInt(FILE* f)

{
    char c;
    int v, sign;

    c = ReadChar(f);
    sign = FALSE;
    v = 0;

    if (c == '-') {
        sign = TRUE;
        c = getc(f);
    }
    while (isdigit(c)) {
        v = 10 * v + (c - '0');
        c = getc(f);
    }
    if (sign)
        v = -1 * v;

    ungetc(c, f);
    return v;
}


int toDot(DdManager* manager, DdNode** f, int num, const char* fp)
{

	const char* inames[] =
	{
		"k0","k1","k2","k3","k4","u0","u1","u2",
		"x0","x1","x2","x3","x4","H0","H1","H2",
		"H3","H4","v0","v1","v2","y0","y1","y2",
		"y3","y4","I0","I1","I2","I3","I4","w0","w1",
		"w2","z0","z1","z2","z3","z4","J0","J1","J2",
		"J3","J4"
	};

	const char* onames[] =
	{
		"f1","f2","f3","f4","f5",
	};

	FILE* file = nullptr;
	if (fopen_s(&file, fp, "w") != 0 || f == NULL || file == NULL)
	{
		printf("can not open to dot.\n");
		return -1;
	}

	Cudd_DumpDot_HK(manager, num, f, inames, onames, file);
	fclose(file);
	return 0;
}

void removeZeroEdges(const char* inPath, const char* outPath) {
	FILE* in = NULL, * out = NULL;
	if (fopen_s(&in, inPath, "r") != 0 || in == NULL) return;
	if (fopen_s(&out, outPath, "w") != 0 || out == NULL) {
		fclose(in); return;
	}

	char zeroId[128] = { 0 };
	char line[1024];


	while (fgets(line, sizeof(line), in)) {
		if (strstr(line, "label = \"0\"")) {

			char* q1 = strchr(line, '\"');
			char* q2 = strchr(q1 + 1, '\"');
			if (q1 && q2 && (q2 - q1 - 1) < sizeof(zeroId)) {
				size_t copyLen = q2 - q1 + 1;
				strncpy_s(zeroId, sizeof(zeroId), q1, copyLen);
				zeroId[copyLen] = '\0';
				break;
			}

		}
	}

	if (strlen(zeroId) == 0) {
		printf("No 0-node found.\n");
		rewind(in);
	}
	else {
		printf("Detected zero node id = %s\n", zeroId);
		rewind(in);
	}


	while (fgets(line, sizeof(line), in)) {
		if (strstr(line, "->") && strstr(line, zeroId))
			continue;
		fputs(line, out);
	}

	fclose(in);
	fclose(out);
}


wchar_t* ConvertToWideChar(const char* str) {
	int len = MultiByteToWideChar(CP_ACP, 0, str, -1, NULL, 0);
	wchar_t* wstr = new wchar_t[len];
	MultiByteToWideChar(CP_ACP, 0, str, -1, wstr, len);
	return wstr;
}

int toJPG(const char* dotPath, const char* jpgPath)
{
	char cmd[512];


	sprintf_s(cmd, sizeof(cmd), "dot -Tjpg \"%s\" -o \"%s\"",
		dotPath, jpgPath);



	int ret = system(cmd);
	if (ret != 0) {
		fprintf(stderr, "Error: dot.exe failed to generate JPG.\n");
		return -1;
	}


	wchar_t* wJpgPath = ConvertToWideChar(jpgPath);
	ShellExecuteW(NULL, L"open", wJpgPath, NULL, NULL, SW_SHOWNORMAL);
	delete[] wJpgPath;
	return 0;
}

void JPG(DdManager* dd, DdNode* f) {
	toDot(dd, &f, 1, "PG.dot");
	removeZeroEdges("PG.dot", "filtered.dot");
	toJPG("filtered.dot", "PG.jpg");
}
