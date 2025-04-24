/*
 * author: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#ifndef _CONFIG_H_
#define _CONFIG_H_

#include <cstddef>

/*
 * Package
 * After a vcpkg update, it can no longer install packages because "vcpkg_cmake" failure
 * Guess it's because of cmake version, not fixed at this stage
 */
// #define ARROW_INSTALLED
// #define EDLIB_INSTALLED


/*
 * IO & tokenzie
 */
#define PARQUET_PREFIX_MIN_LENGTH 4

#define REPORT_TABLE_IN_BUFFER
#define REPORT_TOKEN_IN_BUFFER

// 0: put multiple spaces as a space
// 1: skip all characters not a-z/A-Z/0-9
#define NORMALIZE_STRATEGY 1
// #define STRING_NORMALIZE
#define SKIP_NO_ALPHANUMERIC 0
// #define CHECK_TOKENIZE


/*
 * Parelled
 */
#define MAXTHREADNUM 160

#define MAINTAIN_VALUE 1
#define MAINTAIN_VALUE_OVLP 1
#define MAINTAIN_VALUE_EDIT 0
#define EARLY_TERMINATE 0
#define MAX_PAIR_SIZE 10000000 // for each heap (thread)

#define DEDUPLICATE 1

/*
 * Serial
 */
#define MAX_PAIR_SIZE_SERIAL 1000000000


/*
 * String join
 */
constexpr size_t stringHashNumber = 31;
constexpr size_t modNumber = 1000000007;

#define APPROXIMATE 0
#define TIMER_ON 0
#define VERIFY_PREFIX 0
#define DROP_EMPTY 1
#define LEAVE_EXACT_MATCH 0
#define REPORT_STR_COUNT 0


/*
 * Set join
 */
#define BRUTE_FORCE 1 // we flip the value, that is, 1 for non-bruteforce
#define OUTPUT_DUP 0

// #define WRITE_RESULT

#define PACK(x, y) ((x << 32) + y)
#define PRIME 2017
#define EPS 1e-5
#define NEG -1
#define INF 100000000
#define MAX_LINE_LENGTH 100000
#define CACHE_SIZE 5
#define PART_COE 1

#define APPEND_EMPTY 0
#define MAX_EMPTY_SIZE 1000000
#define RESIZE_DATA 0

// Macro to define the version of the algorithm.
// If VERSION is set to 2, the bottomk variant is used.
#define VERSION 1

// Type alias for token length.
// Use unsigned int for ngram, unsigned short otherwise.
using TokenLen = unsigned int;


/*
 * Overlap join
 */
#define RATIO 0.005
#define TIMES 200

#define BRUTEFORCE_COMB 0
#define PREPROCESS_TIMER_ON 1
#define REPORT_INDEX 0
#define REPORT_BINARY 0
#define REPORT_LIST 0
#define LIMIT_INV_SIZE 1
#define MAX_INV_SIZE 100000
#define APPROXIMATE_OVLP 0
#define SHARING_PREFIX 1


/*
 * sim funcs
 */
#define OVLP_STRATEGY 1


/*
 * Simjoin hpp
 */
#define USING_CRITICAL 0
#define USING_PARALLEL 0
#define MAX_TOTAL_SIZE 1000000000 // exactly the same as MAX_PAIR_SIZE_SERIAL


/*
 * blocker main
 */
#define PRINT_RULES 0
#define EXPORT_MISS 0


/*
 * Miscellaneous
 */
#define VERIFY_JOIN

#define MAX_BITSET_LENGTH 10000000


#endif // _CONFIG_H_