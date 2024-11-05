/*
 * author: Chaoji Zuo and Zhizhi Wang in rutgers-db/SIGMOD2022-Programming-Contest-Public
 * modified: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#ifndef _DATAFRAME_H_
#define _DATAFRAME_H_

#include "common/type.h"
#include <iostream>
#include <fstream>
#include <vector> 
#include <string>
#include <unordered_map>
#include <algorithm>


class Table 
{
public:
	int tid;
	int row_no, col_no;
	std::string table_name;

	std::vector<std::string> schema; // headers
	std::unordered_map<std::string, unsigned int> inverted_schema;
	std::vector<std::vector<std::string>> rows;
	std::vector<std::vector<std::string>> cols;
	std::vector<ui> perfectid; // row id whose all attrs are non-empty

public:
	Table() = default;
	Table(int id, const std::string &name) : tid(id), table_name(name) { }
	Table(int id, const std::string &name, const std::vector<std::string> &data_headers, 
          const std::vector<std::vector<std::string>> &data_rows, 
		  const std::vector<std::vector<std::string>> &data_columns)
    : tid(id), table_name(name), schema(data_headers), rows(data_rows), cols(data_columns) {
        row_no = rows.size();
        col_no = cols.size();
    }

public:
	void Profile();
	void PrintInfo();
	void printData() const;
	void printMetaData(const std::string& filename) const;
	void printGoldData(const std::string& filename, ui tableAsize) const;
	void findPerfectEntity(); // find entities without nan in any attributes
	void insertOneRow(const std::vector<std::string> &tmpRow);
};


struct Rule
{
	std::string attr;
	std::string sim;     
	std::string sim_measure {"none"};      // distance or similarity
	std::string tok {"none"};              // q-gram or dlm 
	std::string tok_settings {"none"};     // # of q or type of dlm
	bool sign;           		           // 0: -/< and 1: +/> 
	double threshold;

	Rule() = default;
	~Rule() = default;
	Rule(const Rule& other) = delete;
	Rule(Rule&& other) = delete;
};


struct Feature
{
	std::string attr;
	std::string sim;     
	std::string sim_measure {"none"};      // distance or similarity
	std::string tok {"none"};              // q-gram or dlm 
	std::string tok_settings {"none"};     // # of q or type of dlm

	Feature() = default;
	~Feature() = default;
	Feature(const Feature& other) = delete;
	Feature(Feature&& other) = delete;
};

#endif // _DATAFRAME_H_