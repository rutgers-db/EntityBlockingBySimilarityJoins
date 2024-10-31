/*
 * author: Chaoji Zuo and Zhizhi Wang in rutgers-db/SIGMOD2022-Programming-Contest-Public
 * modified: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#include "common/dataframe.h"


void Table::Profile() 
{
	row_no = rows.size();
	col_no = cols.size();

	for(ui i = 0; i < schema.size(); i++)
		inverted_schema.insert({schema[i], i});
}


void Table::PrintInfo() 
{
	std::cout << " number of rows: " << row_no << std::endl;
	std::cout << " number of columns: " << col_no << std::endl;
	std::cout << " the schema: ";
	for (auto &attr : schema) std::cout << attr << "; ";
	std::cout << std::endl;
}


void Table::printData() const
{
	for(ui i = 0; i < schema.size(); i++)
		printf("%s\t", schema[i].c_str());
	printf("\n");

	for(ui i = 0; i < 5; i++) {
		for(ui j = 0; j < schema.size(); j++)
			printf("%s\t", cols[j][i].c_str());
		printf("\n");
	}

	for(ui i = 0; i < 5; i++)
		printf("...\t");
	printf("\n");

	for(ui i = rows.size() - 5; i < rows.size(); i++) {
		for(ui j = 0; j < schema.size(); j++)
			printf("%s\t", cols[j][i].c_str());
		printf("\n");
	}
}


void Table::printMetaData(const std::string &filename) const 
{
	FILE* fp = fopen(filename.c_str(), "w");
	if(fp == NULL) {
		printf("Cannot open: %s\n", filename.c_str());
		exit(1);
	}

	ui schema_size = schema.size();
	for(ui i = 0; i < schema_size; i++)
		fprintf(fp, "%s\t", schema[i].c_str());
	fprintf(fp, "\n");

	ui row_size = rows.size();
	for(ui i = 0; i < row_size; i++) {
		for(ui j = 0; j < schema_size; j++)
			fprintf(fp, "%s\t", rows[i][j].c_str());
		fprintf(fp, "\n");
	}

	fclose(fp);
}


void Table::printGoldData(const std::string &filename, ui tableAsize) const 
{
	FILE* fp = fopen(filename.c_str(), "w");
	if(fp == NULL) {
		printf("Cannot open: %s\n", filename.c_str());
		exit(1);
	}

	ui schema_size = schema.size();
	for(ui i = 0; i < schema_size; i++)
		fprintf(fp, "%s\t", schema[i].c_str());
	fprintf(fp, "\n");

	ui row_size = rows.size();
	std::vector<std::vector<ui>> golds;
	golds.resize(tableAsize, std::vector<ui>());
	// for(ui i = 0; i < row_size; i++)
	// 	golds.emplace_back();

	for(ui i = 0; i < row_size; i++) {
		ui idA = atoi(rows[i][0].c_str());
		ui idB = atoi(rows[i][1].c_str());
		golds[idA].emplace_back(idB);
	}
	for(ui i = 0; i < row_size; i++)
		std::sort(golds[i].begin(), golds[i].end());

	for(ui i = 0; i < row_size; i++) {
		ui size = golds[i].size();
		for(ui j = 0; j < size; j++)
			fprintf(fp, "%d %d\n", i, golds[i][j]);
	}

	fclose(fp);
}


void Table::findPerfectEntity()
{
	ui count = 0;
	ui rowid = 0;

	for(const auto &row : rows) {
		bool isPerfect = true;
		for(const auto &attr : row) {
			if(attr.empty()) {
				isPerfect = false;
				break;
			}
		}

		if(isPerfect) {
			++ count;
			perfectid.emplace_back(rowid);
		}

		++ rowid;
	}

	printf("Total perfect entities: %u\n", count);
}