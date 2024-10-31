/*
 * author: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#ifndef _SAMPLER_UTIL_H_
#define _SAMPLER_UTIL_H_

#include "common/io.h"
#include "common/dataframe.h"
#include <vector>
#include <string>


class SamplerUtil
{
public:
	SamplerUtil() = default;
	~SamplerUtil() = default;
	SamplerUtil(const SamplerUtil &other) = delete;
	SamplerUtil(SamplerUtil &&other) = delete;

public:
	static void readCSVTables(const std::string &pathTableA, const std::string &pathTableB, Table &tableA, Table &tableB, 
							  bool isRS);
};


#endif // _SAMPLER_UTIL_H_