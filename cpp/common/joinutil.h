/*
 * author: Dong Deng 
 * modified: Zhencan Peng in rutgers-db/RedPajama_Analysis
 * modified: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#ifndef _JOINUTIL_H_
#define _JOINUTIL_H_

#include "common/type.h"
#include "common/index.h"
#include <chrono>
#include <ios>
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <set>
#include <algorithm>
#include <sys/time.h>
#include <sys/sysinfo.h>
#include <omp.h>
#include <unistd.h>


/*
 * Setjoin
 */
class SetJoinUtil
{
public:
	SetJoinUtil() = default;
	~SetJoinUtil() = default;
	SetJoinUtil(const SetJoinUtil& other) = delete;
	SetJoinUtil(SetJoinUtil&& other) = delete;

public:
	static void printMemory();
	static void processMemUsage(double &vm_usage, double &resident_set);
	// turn on the timer
	static std::chrono::_V2::system_clock::time_point logTime();
	// turn off the timer
	static double repTime(const std::chrono::_V2::system_clock::time_point &start);
};


class SetJoinParallelUtil : public SetJoinUtil
{
public:
	SetJoinParallelUtil() = default;
	~SetJoinParallelUtil() = default;
	SetJoinParallelUtil(const SetJoinParallelUtil& other) = delete;
	SetJoinParallelUtil(SetJoinParallelUtil&& other) = delete;

public:
	static int getHowManyThreads();
	static void printHowManyThreads();
	static std::vector<int> 
	getUniqueInts(const std::vector<std::pair<int, int>>& pairs);
	static void 
	mergeArrays(std::vector<std::vector<std::pair<int,int>>>* input, int arr_len, 
				std::vector<std::vector<std::pair<int,int>>> & result);
	static ui hval(const std::pair<ui, ui> &hf, ui &word);
	static void generateHashFunc(ui seed, std::pair<ui, ui> &hf);
	// bottom K
	static double shrinkBottomk(std::vector<std::vector<ui>>&  bottom_ks, double ratio);
	template<typename T>
	static bool bottomKJaccard(const std::vector<T>& A, const std::vector<T>& B, double& thres);
};


/*
 * String join
 */
class StringJoinUtil
{
public:
	StringJoinUtil() = default;
	~StringJoinUtil() = default;
	StringJoinUtil(const StringJoinUtil& other) = delete;
	StringJoinUtil(StringJoinUtil&& other) = delete;

public:
	static bool strLessT(const std::string &s1, const std::string &s2);
	static uint64_t strHash(const std::string &str, int stPos, int len);
	static int min(int a, int b, int c) {
		return (a <= b && a <= c) ? a : (b <= c ? b : c);
	}
	static int min(int *arr) {
		return arr[std::min_element(arr, arr + 3) - arr];
	}
	static int max(int a, int b, int c);
	static char min(char a, char b, char c);
	static unsigned min(unsigned a, unsigned b, unsigned c);
	static bool PIndexLess(const PIndex &p1, const PIndex &p2);
};

#endif // _JOIN_UTIL_H_