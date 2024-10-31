/*
 * author: Chaoji Zuo and Zhizhi Wang in rutgers-db/SIGMOD2022-Programming-Contest-Public
 * modified: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#include "common/io.h"


/*
 * CSVReader
 */
void CSVReader::strNormalize(std::string &s) 
{
    std::string str = "";
    str.reserve(s.size());
    char prev_char = ' ';

    for (ui i = 0; i < s.size(); i++) {
      if (prev_char == ' ' && s[i] == ' ')
        continue;
      prev_char = s[i];
      str.push_back(tolower(s[i]));
      // if (isalnum(s[i]) || s[i] == '_')  str.push_back(tolower(s[i]));
      // else if (!str.empty() && str.back() != ' ') str.push_back(' ');
    }

    if (!str.empty() && str.back() == ' ') 
		str.pop_back();

    s = str;
}


bool CSVReader::reading_one_table(const std::string &datafilepath, bool normalize) 
{
	int id = 0;
	max_val_len = 0;

	// check file
	if (!ends_with(datafilepath, ".csv")) {
		std::cerr << "WARNING: Skipped non-csv file " << datafilepath << std::endl;
		return false;
	}

	tables.emplace_back(id, datafilepath);
	if (get_table(datafilepath, tables.back().schema, tables.back().cols, tables.back().rows, normalize)) {
		// successfully added, profile and increase the id
		id = id + 1;
		tables.back().Profile();
		// cout << " Added. " << tables.back().schema.size() << " columns and " << tables.back().rows.size() << " rows " << endl;
	} 
	else {
		tables.pop_back();
		perror ("could not open file!!");
		return false;
	}

	return true;
}


void CSVReader::write_one_table(const Table &table, const std::string &outfilename)
{
	std::ofstream out(outfilename, std::ios::out);
	std::string del = "";

	for (auto attribute : table.schema) {
		out << del << attribute;
		del = ",";
	}

	out << std::endl;

	for (auto &row : table.rows) {
		del = "";
		for (auto &val : row) {
			out << del << "\"" << val << "\"";
			del = ",";
		}
		out << std::endl;
	}

	out.close();
}


bool CSVReader::reading(const std::string &datafilespath, bool normalize) 
{
	DIR *dir;
	struct dirent *ent;
	int id = 0;
	max_val_len = 0;

	if ((dir = opendir(datafilespath.c_str())) != NULL) {
		while ((ent = readdir(dir)) != NULL) {
			// check file
			if (!ends_with(ent->d_name, ".csv")) {
				std::cerr << "WARNING: Skipped non-csv file " << ent->d_name << std::endl;
				continue;
			}

			tables.emplace_back(id, std::string(ent->d_name));
			if (get_table(datafilespath + "/" + ent->d_name, tables.back().schema, tables.back().cols, tables.back().rows, normalize)) {
				// successfully added, profile and increase the id
				id = id + 1;
				tables.back().Profile();
				// cout << " Added. " << tables.back().schema.size() << " columns and " << tables.back().rows.size() << " rows " << endl;
			} 
			else {
				tables.pop_back();
			}
		}
		closedir (dir);
	} 
	else {
		perror ("could not open catalog directory!!");
		return false;
	}
	return true;
}


bool CSVReader::get_table(const std::string &filepath, std::vector<std::string> &headers, 
                          std::vector<std::vector<std::string>> &columns, 
                          std::vector<std::vector<std::string>> &rows, 
                          bool normalize) 
{
	std::ifstream in(filepath, std::ios::in);
	if (in.fail()) 
		return (std::cout << "File not found: " + filepath << std::endl) && false;

	csv_read_row(in, headers, normalize);
	columns.resize(headers.size());

	while(in.good()) {
		std::vector<std::string> tuple;
		tuple.reserve(headers.size());
		csv_read_row(in, tuple, normalize);

		if (tuple.size() != headers.size()) {
			std::cout << "Skipped a row" << std::endl;
			continue; // return (cout << "Skipped Broken csv file: " +  filepath << endl) && false;
		}

		// int row_id = rows.size();
		for (ui col = 0; col < tuple.size(); col++) {
			if (tuple[col].empty()) 
				continue;
			if ((int)tuple[col].length() > max_val_len)  
				max_val_len = tuple[col].length();
			// if (columns[col].find(tuple[col]) == columns[col].end())
			// 	columns[col][tuple[col]] = row_id;
		}
		rows.push_back(tuple);

		for(ui i = 0; i < headers.size(); i++)
			columns[i].emplace_back(tuple[i]);
	}
	in.close();
	return true;
}

// **************************************************
// YOU DO NOT HAVE TO CHANGE ANY OF THE FOLLOWING CODE
// **************************************************
// Read in a row and fill in row, normalize the row if isNorm = true
void CSVReader::csv_read_row(std::istream &in, std::vector<std::string> &row, bool isNorm) 
{
	std::stringstream ss;
	bool inquotes = false;
	
	while(in.good()) {
		char c = in.get();
		if (!inquotes && c == '"') //beginquotechar
			inquotes = true;
		//quotechar
		else if (inquotes && c == '"') {
			if (in.peek() == '"') //2 consecutive quotes resolve to 1
				ss << (char)in.get();
			else //endquotechar
				inquotes = false;
		}
		//end of field
		else if (!inquotes && c == field_delimiter) {
			std::string temp_str = ss.str();
			if (isNorm) 
				strNormalize(temp_str);
			row.push_back(temp_str);
			ss.str("");
		}
		else if (!inquotes && (c == '\r' || c == '\n')) {
			if (in.peek() == '\n')  
				in.get();
			std::string temp_str = ss.str();
			if (isNorm) 
				strNormalize(temp_str);
			row.push_back(temp_str);
			return;
		}
		else {
			ss << c;
		}
	}
}


int CSVReader::filesize(const char* filename) 
{
	std::ifstream in(filename, std::ifstream::ate | std::ifstream::binary);
	return in.tellg(); 
}

bool CSVReader::ends_with(std::string const & value, std::string const & ending) 
{
	if (ending.size() > value.size()) return false;
	return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}


void RuleReader::readRules(ui &numRules, Rule *&rules, const std::string &rulePath)
{
	FILE *fp = fopen(rulePath.c_str(), "r");
	if(fp == nullptr) {
		std::cerr << "Cannot open rules file: " << rulePath << std::endl;
		exit(1); 
	}

	fscanf(fp, "%d\n", &numRules);
	rules = new Rule[numRules];

	for(ui i = 0; i < numRules; i++) {
		char ruleStr[200];
		char sign;
		double threshold;
		fscanf(fp, "%s %c %lf", ruleStr, &sign, &threshold);
		// cout << ruleStr << " " << sign << " " << threshold << endl << flush;

		rules[i].sign = (sign == '+');
		rules[i].threshold = threshold;

		// cout << i << " " << flush;

		// extract rule
		char* temp = strtok(ruleStr, "_"); // lattr
		rules[i].attr = std::string(temp);
		temp = strtok(NULL, "_");  // rattr
		temp = strtok(NULL, "_");  // sim
		rules[i].sim = std::string(temp);
		temp = strtok(NULL, "_"); // tok or measure
		// cout << temp << endl << flush;
		if(temp != nullptr) { 
			if(!strcmp(temp, "dist") || !strcmp(temp, "sim")) // lev_dist or lev_sim
				rules[i].sim_measure = std::string(temp);
			else { // tokenizer
				rules[i].tok = std::string(temp);
				temp = strtok(NULL, "_");
				rules[i].tok_settings = std::string(temp);
			}
		}
	}

	fclose(fp);
}

void RuleReader::readFeatureNames(ui &numFeatures, Rule *&rules, 
								  const std::string &defaultFeatureNameDir)
{
	std::string fullPath = __FILE__;
    size_t lastSlash = fullPath.find_last_of("/\\");
    std::string directory = fullPath.substr(0, lastSlash + 1);
    directory = defaultFeatureNameDir == "" ? directory + "../../output/buffer/" 
											: (defaultFeatureNameDir.back() == '/' ? defaultFeatureNameDir 
																				   : defaultFeatureNameDir + "/");
	const std::string pathFeatureName = directory + "feature_names.txt";
	FILE *fp = fopen(pathFeatureName.c_str(), "r");
	if(fp == nullptr) {
		std::cerr << "Cannot open feature name file" << std::endl;
		exit(1); 
	}

	fscanf(fp, "%d\n", &numFeatures);
	rules = new Rule[numFeatures];

	for(ui i = 0; i < numFeatures; i++) {
		char ruleStr[200];
		fscanf(fp, "%s\n", ruleStr);
		// cout << ruleStr << " " << sign << " " << threshold << endl << flush;

		// extract rule
		char* temp = strtok(ruleStr, "_"); // lattr
		rules[i].attr = std::string(temp);
		temp = strtok(NULL, "_");  // rattr
		temp = strtok(NULL, "_");  // sim
		rules[i].sim = std::string(temp);
		temp = strtok(NULL, "_"); // tok or measure
		// cout << temp << endl << flush;
		if(temp != nullptr) { 
			if(!strcmp(temp, "dist") || !strcmp(temp, "sim")) // lev_dist or lev_sim
				rules[i].sim_measure = std::string(temp);
			else { // tokenizer
				rules[i].tok = std::string(temp);
				temp = strtok(NULL, "_");
				rules[i].tok_settings = std::string(temp);
			}
		}
	}

	fclose(fp);
}


void readFile(const char* filename, std::vector<std::vector<int>>& records) 
{
	FILE *fp = fopen(filename, "r");
	fseek(fp, 0, SEEK_END);
	int sz = ftell(fp);
	rewind(fp);
	char *buffer = (char*) malloc(sizeof(char) * (sz + 1));
	sz = fread(buffer, 1, sz, fp);
	buffer[sz] = '\0';

	std::vector<int> rec;
	int num = 0;
	bool empty = true;
	for (char* c = buffer; ; ++c) {
		if (*c == '\0' || isspace(*c)) {
			if (!empty) {
				rec.push_back(num);
				empty = true;
				num = 0;
			}
			if (*c == '\0' || *c == '\n' || *c == '\r') {
				if (!rec.empty()) {
					// lengthSum += rec.size();
					records.push_back(rec);
					rec.clear();
				}
				if (*c == '\0') break;
			}
		} else {
			empty = false;
			printf("%c ", *c);
			num = (num << 3) + (num << 1) + *c - '0';
		}
	}
}



#ifdef ARROW_INSTALLED
/*
 * Parquet Reader
 */
// Private
bool ParquetReader::endWith(const std::string &path, const std::string &ending)
{
	if(ending.size() > path.size())
		return false;
	return std::equal(ending.rbegin(), ending.rend(), path.rbegin());
}


TableType ParquetReader::beginWith(const std::string &path)
{
	if(path.size() < PARQUET_PREFIX_MIN_LENGTH)
		return TableType::Invalid;

	std::string gold_prefix = "gold";
	std::string data_prefix = "table_";
	
	if(std::equal(gold_prefix.begin(), gold_prefix.end(), path.begin()))
		return TableType::Gold;
	else if(std::equal(data_prefix.begin(), data_prefix.end(), path.begin()))
		return TableType::Data;
	else 
		return TableType::Invalid;
}


void ParquetReader::chunkedArray2StringVector(std::shared_ptr<arrow::ChunkedArray> const &array_a, 
										  	  std::vector<std::string> &int64_values)
{
	int64_values.reserve(array_a->length());

	ui num_chunks = array_a->num_chunks();
	for(ui i = 0; i < num_chunks; ++i) {
		auto inner_arr = array_a->chunk(i);
		auto int_a = std::static_pointer_cast<arrow::StringArray>(inner_arr);

		ui int_a_length = int_a->length();
		for(ui j = 0; j < int_a_length; ++j) {
			// std::cout << std::string(int_a->Value(j)) << std::endl;
#ifdef STRING_NORMALIZE
			std::string temp = std::string(int_a->Value(j));
			ParquetReader::stringNormalize(temp);
			int64_values.emplace_back(temp);
			// printf("%s\n", temp.c_str());
#else
			int64_values.emplace_back(std::string(int_a->Value(j)));
#endif
		}
	}
}


template <typename T, typename arrayT>
void ParquetReader::chunkedArray2Vector(std::shared_ptr<arrow::ChunkedArray> const& array_a, 
									    std::vector<T> &values)
{
	values.reserve(array_a->length());

	ui num_chunks = array_a->num_chunks();
	for(ui i = 0; i < num_chunks; ++i) {
		auto inner_arr = array_a->chunk(i);
		auto int_a = std::static_pointer_cast<arrayT>(inner_arr);

		ui int_a_length = int_a->length();
		for(ui j = 0; j < int_a_length; ++j) {
			values.emplace_back(int_a->Value(j));
		}
	}
}


template <typename T>
void ParquetReader::dataVector2TableVector(const std::vector<T> &data_vec, Table& table)
{
	ui value_size = data_vec.size();
	for(ui j = 0; j < value_size; j++) {
#ifdef STRING_NORMALIZE
		std::string temp = std::to_string(data_vec[j]);
		ParquetReader::stringNormalize(temp);
		// printf("%s\n", temp.c_str());
		table.rows[j].emplace_back(temp);
#else
		table.rows[j].emplace_back(std::to_string(data_vec[j]));
#endif
	}
}


std::string ParquetReader::splitSchemaName(const std::string &column_name, const std::string &pattern)
{
	char * strc = new char[strlen(column_name.c_str())+1];
    strcpy(strc, column_name.c_str());   
    char* temp = strtok(strc, pattern.c_str());
	std::string new_colname(temp);
	delete[] strc;
    return new_colname;
}


// Public
void ParquetReader::stringNormalize(std::string &s)
{
	std::string str = "";
    str.reserve(s.size());
    char prev_char = ' ';

    for (ui i = 0; i < s.size(); i++) {
#if NORMALIZE_STRATEGY == 0
		if (prev_char == ' ' && s[i] == ' ')
			continue;
#elif NORMALIZE_STRATEGY == 1
		if(!isalnum(s[i]))
			continue;	
#endif
		prev_char = s[i];
		str.push_back(tolower(s[i]));
    }
    if (!str.empty() && str.back() == ' ') 
		str.pop_back();

    s = str;
}


arrow::Status ParquetReader::readTable(const std::string& filename, Table &table)
{
	printf("%s\n", filename.c_str());

	std::shared_ptr<arrow::io::ReadableFile> infile;
  	ARROW_ASSIGN_OR_RAISE(infile, arrow::io::ReadableFile::Open(filename));

	std::unique_ptr<parquet::arrow::FileReader> reader;
  	PARQUET_THROW_NOT_OK(parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader));

	// Read into the table
	std::shared_ptr<arrow::Table> parquet_table;
  	PARQUET_THROW_NOT_OK(reader->ReadTable(&parquet_table));
	auto const& ptable_ = *parquet_table;
	// arrow::PrettyPrint(ptable_, {}, &std::cerr);
	auto table_array = ptable_.columns();

	ui size = table_array.size();
	ui num_row = table_array[0]->length();
	for(ui i = 0; i < size; i++) {
		if(table_array[i]->length() != num_row) {
			std::cerr << "Broken at column: " << num_row << " " << table_array[i]->length() << std::endl;
			exit(1);
		}
	}
	for(ui i = 0; i < num_row; i++) 
		table.rows.emplace_back();

	for(ui i = 0; i < size; i++) {
		// schema
		std::string column_name = ptable_.field(i)->ToString();
		std::string schema_name = ParquetReader::splitSchemaName(column_name, ":");
		table.schema.emplace_back(schema_name);
		table.inverted_schema.insert({schema_name, i});
		// rows
		// int 64
		if(*table_array[i]->type() == *arrow::int64()) {
			std::vector<int64_t> values;
			ParquetReader::chunkedArray2Vector<int64_t, arrow::Int64Array>(table_array[i], values);
			ParquetReader::dataVector2TableVector<int64_t>(values, table);
			// table.cols.emplace_back(values);
		}
		// string
		else if(*table_array[i]->type() == *arrow::utf8()) {
			std::vector<std::string> values;
			ParquetReader::chunkedArray2StringVector(table_array[i], values);
			for(ui j = 0; j < num_row; j++)
				table.rows[j].emplace_back(values[j]);
			// table.cols.emplace_back(values);
		}
		// double
		else if(*table_array[i]->type() == *arrow::float64()) {
			std::vector<double> values;
			ParquetReader::chunkedArray2Vector<double, arrow::DoubleArray>(table_array[i], values);
			ParquetReader::dataVector2TableVector<double>(values, table);
			// table.cols.emplace_back(values);
		}
		else {
			std::cerr << "No such type: " << *table_array[i]->type() << std::endl;
			exit(1);
		}
		// columns
		table.cols.emplace_back();
		for(ui j = 0; j < num_row; j++)
			table.cols[i].emplace_back(table.rows[j][i]);
	}

	table.Profile();

	return arrow::Status::OK();
}	


arrow::Status ParquetReader::readDirectory(const std::string &dirname, ui &num_tables, 
										   Table &gold, Table &table_A, Table &table_B)
{
	DIR *dir = nullptr;
	struct dirent *ent = nullptr;

	if ((dir = opendir(dirname.c_str())) != nullptr) {
		while ((ent = readdir(dir)) != nullptr) {
			std::string filename = ent->d_name;

			// check file
			if (!ParquetReader::endWith(filename, ".parquet")) {
				printf("Skip: %s\n", filename.c_str());
				continue;
			}	

			TableType ttype = ParquetReader::beginWith(filename);
			arrow::Status rstatus;
			filename = dirname + OSseparator() + filename;

			if(ttype == TableType::Gold) {
				rstatus = ParquetReader::readTable(filename, gold);
			}
			else if(ttype == TableType::Data) {
				rstatus = num_tables == 0 ? ParquetReader::readTable(filename, table_A)
										  : ParquetReader::readTable(filename, table_B);
				num_tables ++;
			}
			else {
				printf("Invalid file\n");
				exit(1);
			}
		}
		closedir (dir);
	}
	else {
		perror ("could not open catalog directory!!");
		exit(1);
	}

	return arrow::Status::OK();
}


void RuleReader::extractRules(char *str, const char *pattern, std::vector<std::string> &res)
{
	char* temp = strtok(str, pattern);
	while(temp != nullptr) {
		res.emplace_back(temp);
		temp = strtok(NULL, pattern);
	}
}


void RuleReader::readRules(const std::string &dirname, ui &num_rules, Rule *&rules)
{
	const char* rule_file_name = "rules.txt";
	DIR* dir = nullptr;
	struct dirent* ent = nullptr;

	if((dir = opendir(dirname.c_str())) != nullptr) {
		while((ent = readdir(dir)) != nullptr) {
			std::string filename = ent->d_name;

			// check file
			if (strcmp(filename.c_str(), rule_file_name)) {
				printf("Skip: %s\n", filename.c_str());
				continue;
			}	

			filename = dirname + OSseparator() + filename;
			FILE* rule_file = fopen(filename.c_str(), "r");
			if(rule_file == nullptr) {
				perror("Cannot open rule file");
				exit(1);
			}
			fscanf(rule_file, "%d\n", &num_rules);
			rules = new Rule[num_rules];

			for(ui i = 0; i < num_rules; i++) {
				char cur_rule[200]; char sign; double threshold;
				fscanf(rule_file, "%s %c %lf\n", cur_rule, &sign, &threshold);

				// sign and threshold
				rules[i].sign = (sign == '+');
				rules[i].threshold = threshold;

				// extract rule
				char* temp = strtok(cur_rule, "_"); // lattr
				rules[i].attr = std::string(temp);
				temp = strtok(NULL, "_");  // rattr
				temp = strtok(NULL, "_");  // sim
				rules[i].sim = std::string(temp);
				if(temp != nullptr) { 
					temp = strtok(NULL, "_");
					if(!strcmp(temp, "dist") || !strcmp(temp, "sim")) // lev_dist or lev_sim
						rules[i].sim_measure = std::string(temp);
					else { // tokenizer
						rules[i].tok = std::string(temp);
						temp = strtok(NULL, "_");
						rules[i].tok_settings = std::string(temp);
					}
				}
			}
		}
	}
	else {
		perror("Cannot open directory\n");
		exit(1);
	}
}
#endif


// writer
const std::vector<std::string> MultiWriter::strAttr = {"title", "titles", "author", "authors", "venue", 
													   "name", "brand", "description", "category"};
const std::vector<std::string> MultiWriter::intAttr = {"year", "id"};
const std::vector<std::string> MultiWriter::floatAttr = {"price"};


void MultiWriter::escapeOneRow(std::string &str)
{
	// escape single double quotes
	size_t pos = str.find("\"");
	bool quoteExist = pos != std::string::npos;
	while(pos != std::string::npos) {
		str.insert(pos, "\"");
		pos = str.find("\"", pos + 2);
	}

	// escape seperators & double quotes
	bool seperatorExist = (str.find(",") != std::string::npos) 
						| (str.find("\r") != std::string::npos)
						| (str.find("\n") != std::string::npos);
	if(quoteExist)
		str = "\"" + str + "\"";
	else if(seperatorExist)
		str = "\"" + str + "\"";
}


void MultiWriter::writeSampleResSnowmanCSV(const std::vector<std::pair<int, int>> &pairs, const std::vector<ui> &idMapA, 
									  	   const std::vector<ui> &idMapB, const std::string &defaultOutputDir)
{
	std::string fullPath = __FILE__;
    size_t lastSlash = fullPath.find_last_of("/\\");
    std::string directory = fullPath.substr(0, lastSlash + 1);
    directory = defaultOutputDir == "" ? directory + "../../output/buffer/" 
									   : (defaultOutputDir.back() == '/' ? defaultOutputDir : defaultOutputDir + "/");
    std::string outputPath = directory + "sample_res.csv";

	FILE *fp = fopen(outputPath.c_str(), "w");
	if(fp == nullptr) {
		std::cerr << "Cannot open" << std::endl;
		exit(1);
	}


    fprintf(fp, "_id,ltable_id,rtable_id\n");

    ui size = pairs.size();
    for(ui i = 0; i < size; i++) 
        fprintf(fp, "%d,%d,%d\n", i, idMapA[pairs[i].first], idMapB[pairs[i].second]);

    fclose(fp);
}


void MultiWriter::writeSampleResMegallenCSV(const std::vector<std::pair<int, int>> &pairs, const std::vector<ui> &idMapA, 
											const std::vector<ui> &idMapB, const Table &tableA, const Table &tableB, 
											const std::vector<int> &label, const std::string &defaultOutputDir)
{
	bool needLabel = label.size() == 0 ? false : true;

	std::string fullPath = __FILE__;
    size_t lastSlash = fullPath.find_last_of("/\\");
    std::string directory = fullPath.substr(0, lastSlash + 1);
    directory = defaultOutputDir == "" ? directory + "../../output/buffer/" 
									   : (defaultOutputDir.back() == '/' ? defaultOutputDir : defaultOutputDir + "/");
    std::string outputPath = directory + "sample_res.csv";

	FILE *csvfile = fopen(outputPath.c_str(), "w");
	if(csvfile == nullptr) {
		std::cerr << "Cannot open" << std::endl;
		exit(1);
	}

	// first three schemas
	fprintf(csvfile, "_id,ltable_id,rtable_id");
	// others
	for(ui i = 1; i < tableA.schema.size(); i++) {
		std::string curAttr = tableA.schema[i];
		std::string newAttr = "ltable_" + curAttr;
		fprintf(csvfile, ",%s", newAttr.c_str());
	}
	for(ui i = 1; i < tableB.schema.size(); i++) {
		std::string curAttr = tableB.schema[i];
		std::string newAttr = "rtable_" + curAttr;
		fprintf(csvfile, ",%s", newAttr.c_str());
	}
	fprintf(csvfile, ",label\n");
	// fprintf(csvfile, "\n");

	// values
	ui fsize = pairs.size();
	int positive = 0;

	for(ui l = 0; l < fsize; l++) {
		int lid = idMapA[pairs[l].first];
		int rid = idMapB[pairs[l].second];
		fprintf(csvfile, "%u,%d,%d", l, lid, rid);

		// out attrs
		for(ui i = 1; i < tableA.schema.size(); i++) {
			std::string curAttr = tableA.rows[lid][i];
			escapeOneRow(curAttr);
			fprintf(csvfile, ",%s", curAttr.c_str());
		}
		for(ui i = 1; i < tableB.schema.size(); i++) {
			std::string curAttr = tableB.rows[rid][i];
			escapeOneRow(curAttr);
			fprintf(csvfile, ",%s", curAttr.c_str());
		}

		if(needLabel) {
			fprintf(csvfile, ",%d", label[l]);
			positive = label[l] == 1 ? positive + 1 : positive;
		}

		fprintf(csvfile, "\n");
	}

	if(needLabel) {
		printf("Positive: %d\tNegative: %d\n", positive, fsize - positive);
	}

	fclose(csvfile);
}


void MultiWriter::writeBlockResSnowmanCSV(const Table &tableA, const std::vector<std::vector<int>> &final_pairs, 
										  const std::string &defaultOutputDir)
{
	std::string fullPath = __FILE__;
    size_t lastSlash = fullPath.find_last_of("/\\");
    std::string directory = fullPath.substr(0, lastSlash + 1);
	directory = defaultOutputDir == "" ? directory + "../../output/blk_res/" 
									   : (defaultOutputDir.back() == '/' ? defaultOutputDir : defaultOutputDir + "/");
    std::string outputPath = directory + "blk_res.csv";
	
	FILE *csvfile = fopen(outputPath.c_str(), "w");
	if(csvfile == nullptr) {
		std::cerr << "Cannot open" << std::endl;
		exit(1);
	}

	fprintf(csvfile, "p1,p2\n");
	for(int i = 0; i < tableA.row_no; i++) {
		for(const auto &id2 : final_pairs[i]) {
			// values
			fprintf(csvfile, "%d,%d\n", (int)i, id2);
		}
	}
	fclose(csvfile);
}


void MultiWriter::writeBlockResMegallenCSV(const Table &tableA, const Table &tableB, ui oneTableSize, 
										   const std::vector<std::vector<int>> &final_pairs, 
										   const std::string &defaultOutputDir)
{
	uint64_t tot = 0;
	for(const auto &fv : final_pairs)
		tot += (uint64_t)fv.size();
	ui numBlkTable = 0, curRow = 0;
	long long l = 0;

	std::string fullPath = __FILE__;
    size_t lastSlash = fullPath.find_last_of("/\\");
    std::string directory = fullPath.substr(0, lastSlash + 1);
   	directory = defaultOutputDir == "" ? directory + "../../output/blk_res/" 
									   : (defaultOutputDir.back() == '/' ? defaultOutputDir : defaultOutputDir + "/");
    std::string outputPath = directory + "blk_res";

	std::string filename = outputPath + std::to_string(numBlkTable) + ".csv";
	FILE *csvfile = fopen(filename.c_str(), "w");
	if(csvfile == nullptr) {
		std::cerr << "Cannot open" << std::endl;
		exit(1);
	}
	// header
	fprintf(csvfile, "_id,ltable_id,rtable_id");
	for(ui i = 1; i < tableA.schema.size(); i++) {
		std::string curAttr = tableA.schema[i];
		std::string newAttr = "ltable_" + curAttr;
		fprintf(csvfile, ",%s", newAttr.c_str());
	}
	for(ui i = 1; i < tableB.schema.size(); i++) {
		std::string curAttr = tableB.schema[i];
		std::string newAttr = "rtable_" + curAttr;
		fprintf(csvfile, ",%s", newAttr.c_str());
	}
	fprintf(csvfile, "\n");

	for(int i = 0; i < tableA.row_no; i++) {
		for(const auto &id2 : final_pairs[i]) {
			// new file
			if(curRow >= (ui)oneTableSize) {
				fclose(csvfile);
				filename = outputPath + std::to_string(++ numBlkTable) + ".csv";
				csvfile = fopen(filename.c_str(), "w");
				if(csvfile == nullptr) {
					std::cerr << "Cannot open" << std::endl;
					exit(1);
				}

				// header
				fprintf(csvfile, "_id,ltable_id,rtable_id");
				for(ui i = 1; i < tableA.schema.size(); i++) {
					std::string curAttr = tableA.schema[i];
					std::string newAttr = "ltable_" + curAttr;
					fprintf(csvfile, ",%s", newAttr.c_str());
				}
				for(ui i = 1; i < tableB.schema.size(); i++) {
					std::string curAttr = tableB.schema[i];
					std::string newAttr = "rtable_" + curAttr;
					fprintf(csvfile, ",%s", newAttr.c_str());
				}
				fprintf(csvfile, "\n");

				curRow = 0;
			}

			// values
			fprintf(csvfile, "%lld,%d,%d", l ++, (int)i, id2);
			for(ui j = 1; j < tableA.schema.size(); j++) {
				std::string curAttr = tableA.rows[i][j];
				escapeOneRow(curAttr);
				fprintf(csvfile, ",%s", curAttr.c_str());
			}
			for(ui j = 1; j < tableB.schema.size(); j++) {
				std::string curAttr = tableB.rows[id2][j];
				escapeOneRow(curAttr);
				fprintf(csvfile, ",%s", curAttr.c_str());
			}

			fprintf(csvfile, "\n");
			++ curRow;
		}
	}

	std::string statOutputPath = directory + "stat.txt";
	FILE *statfile = fopen(statOutputPath.c_str(), "w");
	if(statfile == nullptr) {
		std::cerr << "can not open: "<< statOutputPath << std::endl;
		exit(1);
	}
	fprintf(statfile, "%u %lld", numBlkTable + 1, l);
	fclose(statfile);
}


#ifdef ARROW_INSTALLED
void MultiWriter::setFirstThree(parquet::schema::NodeVector &fields)
{
	fields.push_back(parquet::schema::PrimitiveNode::Make(
		"_id", parquet::Repetition::REQUIRED, parquet::Type::INT64,
		parquet::ConvertedType::UINT_64));
	fields.push_back(parquet::schema::PrimitiveNode::Make(
		"ltable_id", parquet::Repetition::REQUIRED, parquet::Type::INT32,
		parquet::ConvertedType::INT_32));
	fields.push_back(parquet::schema::PrimitiveNode::Make(
		"rtable_id", parquet::Repetition::REQUIRED, parquet::Type::INT32,
		parquet::ConvertedType::INT_32));
}


bool MultiWriter::getField(parquet::schema::NodeVector &fields, string &curAttr, string &newAttr)
{
	if(std::find(strAttr.begin(), strAttr.end(), curAttr) != strAttr.end()) {
		fields.push_back(parquet::schema::PrimitiveNode::Make(
			newAttr, parquet::Repetition::OPTIONAL, parquet::Type::BYTE_ARRAY,
			parquet::ConvertedType::UTF8));
	}
	else if(std::find(intAttr.begin(), intAttr.end(), curAttr) != intAttr.end()) {
		fields.push_back(parquet::schema::PrimitiveNode::Make(
			newAttr, parquet::Repetition::REQUIRED, parquet::Type::INT32,
			parquet::ConvertedType::INT_32));
	}
	else if(std::find(floatAttr.begin(), floatAttr.end(), curAttr) != floatAttr.end()) {
		fields.push_back(parquet::schema::PrimitiveNode::Make(
			newAttr, parquet::Repetition::REQUIRED, parquet::Type::DOUBLE,
			parquet::ConvertedType::NONE));
	}
	else {
		std::cerr << "No such attr: " << curAttr << std::endl;
		return false;
	}

	return true;
}


void MultiWriter::writeSampleResSnowmanParquet(const std::vector<std::pair<int, int>> &pairs, const std::vector<ui> &idMapA, 
									  	 	   const std::vector<ui> &idMapB, const std::string &defaultOutputDir)
{
	std::cerr << "called method has not been implemented" << std::endl;
	exit(1);
}


void MultiWriter::writeSampleResMegallenParquet(const std::vector<std::pair<int, int>> &pairs, const std::vector<ui> &idMapA, 
												const std::vector<ui> &idMapB, const Table &tableA, const Table &tableB, 
												const std::vector<int> &label, const std::string &defaultOutputDir)
{
	bool needLabel = label.size() == 0 ? false : true;

	shared_ptr<arrow::io::FileOutputStream> outfile;

	std::string fullPath = __FILE__;
    size_t lastSlash = fullPath.find_last_of("/\\");
    std::string directory = fullPath.substr(0, lastSlash + 1);
    directory = defaultOutputDir == "" ? directory + "../../output/blk_res/" 
									   : (defaultOutputDir.back() == '/' ? defaultOutputDir : defaultOutputDir + "/");
    std::string outputPath = directory + "sample_res.parquet";

	PARQUET_ASSIGN_OR_THROW(
		outfile,
		arrow::io::FileOutputStream::Open(outputPath)
	);

	parquet::WriterProperties::Builder builder;

	// build schema
	parquet::schema::NodeVector fields;
	setFirstThree(fields);

	for(ui i = 1; i < tableA.schema.size(); i++) {
		std::string curAttr = tableA.schema[i];
		std::string newAttr = "ltable_" + curAttr;
		if(!getField(fields, curAttr, newAttr)) {
			cerr << "No such attr: " << curAttr << endl;
			exit(1);
		}
	}
	for(ui i = 1; i < tableB.schema.size(); i++) {
		std::string curAttr = tableB.schema[i];
		std::string newAttr = "rtable_" + curAttr;
		if(!getField(fields, curAttr, newAttr)) {
			cerr << "No such attr: " << curAttr << endl;
			exit(1);
		}
	}

	if(needLabel) {
		fields.push_back(parquet::schema::PrimitiveNode::Make(
			"label", parquet::Repetition::REQUIRED, parquet::Type::INT32,
			parquet::ConvertedType::INT_32));
	}

	auto schema = std::static_pointer_cast<parquet::schema::GroupNode>(
    	parquet::schema::GroupNode::Make("schema", parquet::Repetition::REQUIRED, fields));

	// stream write
	parquet::StreamWriter os{
    	parquet::ParquetFileWriter::Open(outfile, schema, builder.build())};

	ui fsize = pairs.size();
	int positive = 0;

	for(ui l = 0; l < fsize; l++) {
		int lid = idMapA[pairs[l].first];
		int rid = idMapB[pairs[l].second];
		os << uint64_t(l) << lid << rid;

		// out attrs
		for(ui i = 1; i < tableA.schema.size(); i++) {
			std::string curAttr = tableA.schema[i];
			if(std::find(strAttr.begin(), strAttr.end(), curAttr) != strAttr.end()) 
				os << tableA.rows[lid][i];
			else if(std::find(intAttr.begin(), intAttr.end(), curAttr) != intAttr.end()) {
				if(tableA.rows[lid][i] == "")
					os << 0;
				else
					os << std::stoi(tableA.rows[lid][i]);
			}
			else if(std::find(floatAttr.begin(), floatAttr.end(), curAttr) != floatAttr.end()) {
				if(tableA.rows[lid][i] == "")
					os << 0.0;
				else
					os << std::stod(tableA.rows[lid][i]);
			}
		}
		for(ui i = 1; i < tableB.schema.size(); i++) {
			std::string curAttr = tableB.schema[i];
			if(std::find(strAttr.begin(), strAttr.end(), curAttr) != strAttr.end()) 
				os << tableB.rows[rid][i];
			else if(std::find(intAttr.begin(), intAttr.end(), curAttr) != intAttr.end()) {
				if(tableB.rows[rid][i] == "")
					os << 0;
				else
					os << std::stoi(tableB.rows[rid][i]);
			}
			else if(std::find(floatAttr.begin(), floatAttr.end(), curAttr) != floatAttr.end()) {
				if(tableB.rows[rid][i] == "")
					os << 0.0;
				else
					os << std::stod(tableB.rows[rid][i]);
			}
		}

		if(needLabel) {
			os << label[l];
			positive = label[l] == 1 ? positive + 1 : positive;
		}

		os << parquet::EndRow;
	}

	if(needLabel) {
		printf("Positive: %d\tNegative: %d\n", positive, fsize - positive);
	}
}


void MultiWriter::writeBlockResSnowmanParquet(const Table &tableA, const std::vector<std::vector<int>> &final_pairs, 
											  const std::string &defaultOutputDir)
{
	std::cerr << "called method has not been implemented" << std::endl;
	exit(1);
}


void MultiWriter::writeBlockResMegallenParquet(const Table &tableA, const Table &tableB, ui oneTableSize, 
											   const std::vector<std::vector<int>> &final_pairs, 
											   const std::string &defaultOutputDir)
{
	std::cerr << "called method has not been implemented" << std::endl;
	exit(1);
}

#endif 