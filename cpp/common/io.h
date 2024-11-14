/*
 * author: Chaoji Zuo and Zhizhi Wang in rutgers-db/SIGMOD2022-Programming-Contest-Public
 * modified: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#ifndef _IO_H_
#define _IO_H_

#include "common/dataframe.h"
#include "common/type.h"
#include "common/config.h"
#include <iostream>
#include <istream>
#include <sstream>
#include <cstring>
#include <dirent.h>
#include <sys/stat.h>
#include <string.h>
#include <stdio.h>
#include <vector>
#include <codecvt>
#ifdef ARROW_INSTALLED
#include <arrow/compute/api.h>
#include "arrow/pretty_print.h"
#include <arrow/api.h>
#include <arrow/csv/api.h>
#include <arrow/json/api.h>
#include <arrow/io/api.h>
#include <arrow/table.h>
#include <arrow/chunked_array.h>
#include <arrow/pretty_print.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/ipc/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <parquet/exception.h>
#include "arrow/io/file.h"
#include "parquet/stream_writer.h"
#endif


// read
class CSVReader 
{
private:
    int max_val_len = 0;
    char field_delimiter = ','; // '\t'

    int filesize(const char* filename);
    bool ends_with(std::string const & value, std::string const & ending);

    void csv_read_row(std::istream &in, std::vector<std::string> &row, bool isNorm = true);
	void csv_read_chinese_row(std::wistream &in, std::vector<std::wstring> &row, bool isNorm = true);

    bool get_table(const std::string &filepath, std::vector<std::string> &headers, 
                   std::vector<std::vector<std::string>> &columns, 
                   std::vector<std::vector<std::string>> &rows, 
                   bool normalize);
	bool get_chinese_table(const std::string &filepath, std::vector<std::wstring> &headers, 
						   std::vector<std::vector<std::wstring>> &columns, 
						   std::vector<std::vector<std::wstring>> &rows, 
						   bool normalize);

public:
    CSVReader() = default;
    ~CSVReader() = default;
    std::vector<Table> tables;
	std::vector<ChineseTable> chineseTables;

    void strNormalize(std::string &s); // also for the use of query normalization
	void strNormalize(std::wstring &ws);

    bool reading(const std::string &datafilespath, bool normalize);
    void write_one_table(const Table &table, const std::string &outfilename);

    bool reading_one_table(const std::string &datafilepath, bool normalize);
	bool reading_one_chinese_table(const std::string &datafilepath, bool normalize);

    int get_max_val_len() { return max_val_len; };
};


class RuleReader
{
public:
	RuleReader() = default;
	~RuleReader() = default;
	RuleReader(const RuleReader& other) = delete;
	RuleReader(RuleReader&& other) = delete;

public:
	static void readRules(ui &numRules, Rule *&rules, const std::string &rulePath);
	static void readFeatureNames(ui &numFeatures, Rule *&rules, 
								 const std::string &defaultFeatureNameDir = "");

	// unused
	static void extractRules(char* str, const char* pattern, std::vector<std::string>& res);
	static void readRules(const std::string& dirname, ui& num_rules, Rule*& rules);
};


void readFile(const char* filename, std::vector<std::vector<int>>& records);


#ifdef ARROW_INSTALLED
class ParquetReader
{
public:     
    ParquetReader() = default;
    ~ParquetReader() = default;
    ParquetReader(const ParquetReader& other) = delete;
    ParquetReader(ParquetReader&& other) = delete;

private:
	static bool endWith(const std::string& path, const std::string& ending);
	static TableType beginWith(const std::string& path);
	static void chunkedArray2StringVector(std::shared_ptr<arrow::ChunkedArray> const& array_a, 
										  std::vector<std::string>& int64_values);
	template <typename T, typename arrayT>
	static void chunkedArray2Vector(std::shared_ptr<arrow::ChunkedArray> const& array_a, 
								    std::vector<T>& values);
	template <typename T>
	static void dataVector2TableVector(const std::vector<T>& data_vec, Table& table);
	static std::string splitSchemaName(const std::string& column_name, const std::string& pattern);

public:
	// String normalization: Lowercase & Skip spaces.
	static void stringNormalize(std::string& s);
	// Read
	static arrow::Status readTable(const std::string& filename, Table& table);
	static arrow::Status readDirectory(const std::string& dirname, ui& num_table,
									   Table& gold, Table& table_A, Table& table_B);
};
#endif


/*
 * writer
 * for writing blocking / matching / sample results
 * csv & parquet
 * Snowman format & Megallen format
 * the block res is stored by chunking the "big table" to several small tables in "blktmp"
 * the sample res is stored in "buffer"
 */
class MultiWriter
{
private:
	static const std::vector<std::string> strAttr;
	static const std::vector<std::string> intAttr;
	static const std::vector<std::string> floatAttr;

public:
	MultiWriter() = default;
	~MultiWriter() = default;
	MultiWriter(const MultiWriter &other) = delete;
	MultiWriter(MultiWriter &&other) = delete;

	/* 
	 * csv
	 */
public:
	// if seperator found (,) in str, surround it with ""
	// if double quotes found, escaped it with another double quote
	// other chars like '\n' or '<' are common in secret datasets
	// but we can leave them scine pandas praser will handle them
	static void escapeOneRow(std::string &str);

public:
	static void writeOneTable(const Table &table, const std::string &outputFilePath);

	// sample csv
	static void writeSampleResSnowmanCSV(const std::vector<std::pair<int, int>> &pairs, const std::vector<ui> &idMapA, 
									  	 const std::vector<ui> &idMapB, const std::string &defaultOutputDir = "");
	static void writeSampleResMegallenCSV(const std::vector<std::pair<int, int>> &pairs, const std::vector<ui> &idMapA, 
										  const std::vector<ui> &idMapB, const Table &tableA, const Table &tableB, 
										  const std::vector<int> &label, const std::string &defaultOutputDir = "");
	
	// block csv
	static void writeBlockResSnowmanCSV(const Table &tableA, const std::vector<std::vector<int>> &final_pairs, 
										const std::string &defaultOutputDir = "");
	static void writeBlockResMegallenCSV(const Table &tableA, const Table &tableB, ui oneTableSize, 
										 const std::vector<std::vector<int>> &final_pairs, 
										 const std::string &defaultOutputDir = "");

	/* 
	 * parquet
	 */
#ifdef ARROW_INSTALLED
private:
	// utils
	static void setFirstThree(parquet::schema::NodeVector &fields);
	static bool getField(parquet::schema::NodeVector &fields, string &curAttr, string &newAttr);

public:
	// sample parquet
	// unimplemented
	static void writeSampleResSnowmanParquet(const std::vector<std::pair<int, int>> &pairs, const std::vector<ui> &idMapA, 
									  	 	 const std::vector<ui> &idMapB, const std::string &defaultOutputDir = "");
	// avaiable
	static void writeSampleResMegallenParquet(const std::vector<std::pair<int, int>> &pairs, const std::vector<ui> &idMapA, 
										  	  const std::vector<ui> &idMapB, const Table &tableA, const Table &tableB, 
										  	  const std::vector<int> &label, const std::string &defaultOutputDir = "");
	
	// block parquet
	// unimplemented
	static void writeBlockResSnowmanParquet(const Table &tableA, const std::vector<std::vector<int>> &final_pairs, 
											const std::string &defaultOutputDir = "");
	static void writeBlockResMegallenParquet(const Table &tableA, const Table &tableB, ui oneTableSize, 
											 const std::vector<std::vector<int>> &final_pairs, 
											 const std::string &defaultOutputDir = "");
#endif
};

#endif // _IO_H_