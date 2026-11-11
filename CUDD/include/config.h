// config.h - 手动伪造 configure 结果

#ifndef CUDD_CONFIG_H
#define CUDD_CONFIG_H

// 标准头文件存在
#define HAVE_ASSERT_H 1
#define HAVE_STRING_H 1
#define HAVE_STDLIB_H 1
#define HAVE_STDARG_H 1
#define HAVE_LIMITS_H 1
#define HAVE_STDINT_H 1
#define HAVE_TIME_H 1
#define HAVE_INTTYPES_H 1
#define HAVE_POWL 1
#define PACKAGE_VERSION "3.0.0"


// 用不到 UNIX 功能的设为 0
#define HAVE_SYS_RESOURCE_H 0
#define HAVE_SYS_TIME_H 0
#define HAVE_SYS_TIMES_H 0
#define HAVE_UNISTD_H 0
#define HAVE_GETRUSAGE 0
#define HAVE_TIMES 0
#define HAVE_CLOCK 1

// Windows 下没有 fork 等 UNIX API
#define HAVE_FORK 0
#define HAVE_PIPE 0

// snprintf/sprintf 等安全函数
#define HAVE_SNPRINTF 1

// 启用 DDDMP 支持
#define HAVE_DDDMP 1

#endif /* CUDD_CONFIG_H */
