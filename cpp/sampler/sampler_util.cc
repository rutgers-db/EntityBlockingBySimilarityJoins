/*
 * author: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#include "sampler/sampler_util.h"


void SamplerUtil::readCSVTables(const std::string &pathTableA, const std::string &pathTableB, 
							    Table &tableA, Table &tableB, bool isRS)
{
	CSVReader reader;

	bool normalize = false;

	reader.reading_one_table(pathTableA, normalize);
	tableA = reader.tables[0];
	tableA.findPerfectEntity();
	if(isRS == true) {
		reader.reading_one_table(pathTableB, normalize);
		tableB = reader.tables[1];
		tableB.findPerfectEntity();
	}
	else {
		tableB = tableA;
	}

	tableA.Profile();
	tableB.Profile();
}