/*
 * author: Chaoji Zuo and Zhizhi Wang in rutgers-db/SIGMOD2022-Programming-Contest-Public
 * modified: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#include "common/tokenizer.h"

// check
bool Tokenizer::isNotAlphaNumeric(char c)
{
	return isalnum(c) ? false : true;
}

bool Tokenizer::isAlphaNumeric(const std::string &s)
{
	// find_id finds the first iter who makes 'isNotAlphaNumeric' true
	// if no such iter, then s is alphanumeric
	return std::find_if(s.begin(), s.end(), isNotAlphaNumeric) == s.end();
}

// dlm
void Tokenizer::string2TokensDlm(const std::string &s, std::vector<std::string> &res,
								 const std::string &delims)
{
	std::string::size_type begIdx, endIdx;
	begIdx = s.find_first_not_of(delims);

	while (begIdx != std::string::npos)
	{
		endIdx = s.find_first_of(delims, begIdx);
		if (endIdx == std::string::npos)
			endIdx = s.length();
#if SKIP_NO_ALPHANUMERIC == 1
		auto temp = s.substr(begIdx, endIdx - begIdx);
		if (!isAlphaNumeric(temp))
		{
			begIdx = s.find_first_not_of(delims, endIdx);
			continue;
		}
		res.emplace_back(temp);
#else
		res.push_back(s.substr(begIdx, endIdx - begIdx));
#endif
		begIdx = s.find_first_not_of(delims, endIdx);
	}
}

// q-gram
void Tokenizer::string2TokensQGram(const std::string &s, std::vector<std::string> &res,
								   ui q)
{
	// add prefix & suffix
	std::string prefix = "";
	std::string suffix = "";
	for (ui i = 0; i < q - 1; i++)
	{
		prefix += "#";
		suffix += "$";
	}
	std::string str = prefix + s + suffix;

	// tokenize
#if NORMALIZE_STRATEGY == 0
	bool f = true;
	for (auto sit = str.begin(); sit != str.end();)
	{
		if (*sit == ' ' && f == true)
		{
			sit = str.erase(sit);
			continue;
		}
		else if (f == true)
			f = false;
		else
			f = true;
		++sit;
	}
#endif
	for (int i = 0; i < (int)str.length() - (int)q + 1; i++)
	{
		auto temp = str.substr(i, q);
		if (!isAlphaNumeric(temp))
			continue;
		res.emplace_back(temp);
	}
}

// wspace
// a wrapper of space-delimter dlm tokenizer
void Tokenizer::string2TokensWSpace(const std::string &s, std::vector<std::string> &res)
{
	std::string delim = " ";
	Tokenizer::string2TokensDlm(s, res, delim);
}

// alphanumeric
void Tokenizer::string2TokensAlphaNumeric(const std::string &s, std::vector<std::string> &res)
{
	std::regex regexp("[a-zA-Z0-9]+");
	std::sregex_iterator it(s.begin(), s.end(), regexp);
	std::sregex_iterator end;

	while (it != end)
	{
		res.emplace_back(it->str());
		++it;
	}
}

// table private
void Tokenizer::stringNormalize(std::string &s, ui strategy)
{
	std::string str = "";
	str.reserve(s.size());
	char prev_char = ' ';

	for (ui i = 0; i < s.size(); i++)
	{
		if (!strategy)
		{
			if (prev_char == ' ' && s[i] == ' ')
				continue;
		}
		else if (strategy == 1)
		{
			if (!isalnum(s[i]))
				continue;
		}
		else if (strategy == 2)
		{
			if (!isalnum(s[i]) && (s[i] != ' ' || prev_char != ' '))
				continue;
		}

		prev_char = s[i];
		str.push_back(tolower(s[i]));
	}
	if (!str.empty() && str.back() == ' ')
		str.pop_back();

	s = str;
}

void Tokenizer::updateBagDlm(const Table &table, std::vector<std::vector<std::string>> &bow,
							 ui column, const std::string &dlim, ui strategy)
{
	ui num_row = table.rows.size();
	// std::string delim = " ,-_;.?!|()\\\t\n\r";

	for (ui i = 0; i < num_row; i++)
	{
		auto &row = table.rows[i];
		// Skip id
		std::string str(row[column]);
		// Tokenizer::stringNormalize(str, strategy);
		std::vector<std::string> bag_of_words;
		Tokenizer::string2TokensDlm(str, bag_of_words, dlim);
		// Sort
		std::sort(bag_of_words.begin(), bag_of_words.end());
		// Unique
		auto uniq_it = unique(bag_of_words.begin(), bag_of_words.end());
		bag_of_words.resize(distance(bag_of_words.begin(), uniq_it));
		// Update
		for (auto str : bag_of_words)
			bow[i].emplace_back(str);
	}
}

void Tokenizer::updateBagQGram(const Table &table, std::vector<std::vector<std::string>> &bow,
							   ui column, ui q)
{
	ui num_row = table.rows.size();
	for (ui i = 0; i < num_row; i++)
	{
		auto &row = table.rows[i];
		// Skip id
		std::string str = row[column];
		// Tokenizer::stringNormalize(str, 1);
		std::vector<std::string> bag_of_words;
		Tokenizer::string2TokensQGram(str, bag_of_words, q);
		// Sort
		std::sort(bag_of_words.begin(), bag_of_words.end());
		// Unique
		auto uniq_it = unique(bag_of_words.begin(), bag_of_words.end());
		bag_of_words.resize(distance(bag_of_words.begin(), uniq_it));
		// Update
		for (auto str : bag_of_words)
			bow[i].emplace_back(str);
	}
}

void Tokenizer::updateBagAlphaNumeric(const Table &table, std::vector<std::vector<std::string>> &bow,
									  ui column)
{
	ui num_row = table.rows.size();
	for (ui i = 0; i < num_row; i++)
	{
		auto &row = table.rows[i];
		// Skip id
		std::string str = row[column];
		// Tokenizer::stringNormalize(str, 1);
		std::vector<std::string> bag_of_words;
		Tokenizer::string2TokensAlphaNumeric(str, bag_of_words);
		// Sort
		std::sort(bag_of_words.begin(), bag_of_words.end());
		// Unique
		auto uniq_it = unique(bag_of_words.begin(), bag_of_words.end());
		bag_of_words.resize(distance(bag_of_words.begin(), uniq_it));
		// Update
		for (auto str : bag_of_words)
			bow[i].emplace_back(str);
	}
}

void Tokenizer::sortIdMap(std::vector<ui> &id_map, const std::vector<std::vector<ui>> &datasets)
{
	std::sort(id_map.begin(), id_map.end(), [&datasets](const int &id1, const int &id2)
			  {
		int dsize1 = datasets[id1].size();
		int dsize2 = datasets[id2].size();
		if (dsize1 < dsize2)
			return true;
		else if (dsize1 > dsize2)
			return false;
		else {
			for (int i = 0; i < dsize1; i++)
			{
				if (datasets[id1][i] < datasets[id2][i])
					return true;
				else if (datasets[id1][i] > datasets[id2][i])  
					return false;
			}
			if (id1 < id2)
				return true;
			else
				return false;
		} });
}

void Tokenizer::RStableAttr2IntVector(const Table &tableA, const Table &tableB,
									  std::vector<std::vector<ui>> &recordsA,
									  std::vector<std::vector<ui>> &recordsB,
									  std::vector<double> &weightsA,
									  std::vector<double> &weightsB,
									  std::vector<double> &wordwt,
									  std::vector<ui> &id_mapA,
									  std::vector<ui> &id_mapB,
									  ui columnA, ui columnB,
									  TokenizerType tok_type, ui &num_word, ui q)
{
	ui num_rowA = tableA.rows.size();
	ui num_rowB = tableB.rows.size();

	std::unordered_map<std::string, std::vector<std::pair<ui, ui>>> inv_index;
	std::vector<std::vector<std::string>> bowsA;
	std::vector<std::vector<std::string>> bowsB;
	std::vector<std::pair<ui, std::string>> tokens;

	bowsA.resize(num_rowA);
	bowsB.resize(num_rowB);

	switch (tok_type)
	{
	case TokenizerType::Dlm:
	{
		// std::string delim = " ,-_:;.?!|()[]\\\t\n\r";
		// std::string delim = " .,\\\t\r\n";
		std::string delim = " \"\',\\\t\r\n";
		Tokenizer::updateBagDlm(tableA, bowsA, columnA, delim, 0);
		Tokenizer::updateBagDlm(tableB, bowsB, columnB, delim, 0);
		break;
	}
	case TokenizerType::QGram:
	{
		Tokenizer::updateBagQGram(tableA, bowsA, columnA, q);
		Tokenizer::updateBagQGram(tableB, bowsB, columnB, q);
		break;
	}
	case TokenizerType::WSpace:
	{
		std::string delim = " ";
		Tokenizer::updateBagDlm(tableA, bowsA, columnA, delim, 1);
		Tokenizer::updateBagDlm(tableB, bowsB, columnB, delim, 1);
		break;
	}
	case TokenizerType::AlphaNumeric:
	{
		Tokenizer::updateBagAlphaNumeric(tableA, bowsA, columnA);
		Tokenizer::updateBagAlphaNumeric(tableB, bowsB, columnB);
		break;
	}

	default:
	{
		printf("No such tokenizers\n");
		exit(1);
	}
	}

	printf("Building tokens index....\n");
	fflush(stdout);

	// Update index & tokens
	for (ui i = 0; i < num_rowA; i++)
	{
		for (auto &index_str : bowsA[i])
		{
			auto &str = index_str; // token
#if SKIP_NO_ALPHANUMERIC == 1
			if (str.empty() || str == " ")
				continue;
#endif
			inv_index[str].emplace_back(i, 0);
		}
	}
	for (ui i = 0; i < num_rowB; i++)
	{
		for (auto &index_str : bowsB[i])
		{
			auto &str = index_str; // token
#if SKIP_NO_ALPHANUMERIC == 1
			if (str.empty() || str == " ")
				continue;
#endif
			inv_index[str].emplace_back(i, 1);
		}
	}
	num_word = inv_index.size();
	for (auto &entry : inv_index)
		tokens.emplace_back(entry.second.size(), entry.first);

	// Sort according to the token's frequency
	std::sort(tokens.begin(), tokens.end(),
			  [](const std::pair<int, std::string> &p1, const std::pair<int, std::string> &p2)
			  {
				  return p1.first < p2.first;
			  });

#ifdef CHECK_TOKENIZE
	for (ui i = 0; i < tokens.size(); i++)
		all_index[i] = tokens[i].second;
#endif

	// Update records
	std::vector<std::vector<ui>> datasetsA(num_rowA);
	std::vector<std::vector<ui>> datasetsB(num_rowB);

	ui num_tokens = tokens.size();
	ui rec_num = num_rowA + num_rowB;
	for (ui i = 0; i < num_tokens; i++)
	{
		auto &word = tokens[i].second;
		for (auto &j : inv_index[word])
		{
			if (!j.second) // A
				datasetsA[j.first].emplace_back(i);
			else // B
				datasetsB[j.first].emplace_back(i);
		}

		// word weight
		ui freq = tokens[i].first;
		wordwt.emplace_back(log10(rec_num * 1.0 / freq));
	}

	// for(ui i = 0; i < num_tokens; i++) {
	// 	ui freq = tokens[i].first;
	// 	weights.emplace_back(log10(num_row * 1.0 / freq));
	// }

	id_mapA.clear();
	id_mapB.clear();
	for (ui i = 0; i < num_rowA; i++)
		id_mapA.emplace_back(i);
	for (ui i = 0; i < num_rowB; i++)
		id_mapB.emplace_back(i);

	// Sort dataset by size first, element second, id third
	Tokenizer::sortIdMap(id_mapA, datasetsA);
	Tokenizer::sortIdMap(id_mapB, datasetsB);

	for (ui i = 0; i < id_mapA.size(); i++)
		recordsA.emplace_back(datasetsA[id_mapA[i]]);
	for (ui i = 0; i < id_mapB.size(); i++)
		recordsB.emplace_back(datasetsB[id_mapB[i]]);

	// record weight
	weightsA.resize(num_rowA, 0.0);
	weightsB.resize(num_rowB, 0.0);

	for (ui i = 0; i < num_rowA; i++)
		for (const auto &w : recordsA[i])
			weightsA[i] += wordwt[w];
	for (ui i = 0; i < num_rowB; i++)
		for (const auto &w : recordsB[i])
			weightsB[i] += wordwt[w];

			// Remember to disable sort before checking
#ifdef CHECK_TOKENIZE
			// FILE* fpCheck = fopen("./buffer/check_tokens.txt", "w");
			// for(ui i = 0; i < num_row; i++) {
			// 	for(ui j = 0; j < num_col; j++) {
			// 		for(ui k = offsets[i][j]; k < offsets[i][j + 1]; k++)
			// 			fprintf(fpCheck, "%s ", all_index[records[i][k]].c_str());
			// 		fprintf(fpCheck, "\t");
			// 	}
			// 	fprintf(fpCheck, "\n");
			// }
			// fclose(fpCheck);
#endif
}

void Tokenizer::SelftableAttr2IntVector(const Table &tableA,
										std::vector<std::vector<ui>> &recordsA,
										std::vector<double> &weightsA,
										std::vector<double> &wordwt,
										std::vector<ui> &id_mapA,
										ui columnA, TokenizerType tok_type,
										ui &num_word, ui q)
{
	ui num_rowA = tableA.rows.size();

	std::unordered_map<std::string, std::vector<std::pair<ui, ui>>> inv_index;
	std::vector<std::vector<std::string>> bowsA;
	std::vector<std::pair<ui, std::string>> tokens;

	bowsA.resize(num_rowA);

	switch (tok_type)
	{
	case TokenizerType::Dlm:
	{
		// std::string delim = " ,-_:;.?!|()[]\\\t\n\r";
		// std::string delim = " .,\\\t\r\n";
		std::string delim = " \"\',\\\t\r\n";
		Tokenizer::updateBagDlm(tableA, bowsA, columnA, delim, 0);
		break;
	}
	case TokenizerType::QGram:
	{
		Tokenizer::updateBagQGram(tableA, bowsA, columnA, q);
		break;
	}
	case TokenizerType::WSpace:
	{
		std::string delim = " ";
		Tokenizer::updateBagDlm(tableA, bowsA, columnA, delim, 1);
		break;
	}

	default:
	{
		printf("No such tokenizers\n");
		exit(1);
	}
	}

	printf("Building tokens index....\n");
	fflush(stdout);

	// Update index & tokens
	for (ui i = 0; i < num_rowA; i++)
	{
		for (auto &index_str : bowsA[i])
		{
			auto &str = index_str; // token
			inv_index[str].emplace_back(i, 0);
		}
	}
	num_word = inv_index.size();
	for (auto &entry : inv_index)
		tokens.emplace_back(entry.second.size(), entry.first);

	// Sort according to the token's frequency
	std::sort(tokens.begin(), tokens.end(),
			  [](const std::pair<int, std::string> &p1, const std::pair<int, std::string> &p2)
			  {
				  return p1.first < p2.first;
			  });

#ifdef CHECK_TOKENIZE
	for (ui i = 0; i < tokens.size(); i++)
		all_index[i] = tokens[i].second;
#endif

	// Update records
	std::vector<std::vector<ui>> datasetsA(num_rowA);

	ui num_tokens = tokens.size();
	for (ui i = 0; i < num_tokens; i++)
	{
		auto &word = tokens[i].second;
		for (auto &j : inv_index[word])
		{
			if (!j.second) // A
				datasetsA[j.first].emplace_back(i);
		}

		// word weight
		ui freq = tokens[i].first;
		wordwt.emplace_back(log10(num_rowA * 1.0 / freq));
	}

	// for(ui i = 0; i < num_tokens; i++) {
	// 	ui freq = tokens[i].first;
	// 	weights.emplace_back(log10(num_row * 1.0 / freq));
	// }

	id_mapA.clear();
	for (ui i = 0; i < num_rowA; i++)
		id_mapA.emplace_back(i);

	// Sort dataset by size first, element second, id third
	Tokenizer::sortIdMap(id_mapA, datasetsA);

	for (ui i = 0; i < id_mapA.size(); i++)
		recordsA.emplace_back(datasetsA[id_mapA[i]]);

	// record weight
	weightsA.resize(num_rowA, 0.0);

	for (ui i = 0; i < num_rowA; i++)
		for (const auto &w : recordsA[i])
			weightsA[i] += wordwt[w];

			// Remember to disable sort before checking
#ifdef CHECK_TOKENIZE
			// FILE* fpCheck = fopen("./buffer/check_tokens.txt", "w");
			// for(ui i = 0; i < num_row; i++) {
			// 	for(ui j = 0; j < num_col; j++) {
			// 		for(ui k = offsets[i][j]; k < offsets[i][j + 1]; k++)
			// 			fprintf(fpCheck, "%s ", all_index[records[i][k]].c_str());
			// 		fprintf(fpCheck, "\t");
			// 	}
			// 	fprintf(fpCheck, "\n");
			// }
			// fclose(fpCheck);
#endif
}

void Tokenizer::resTableAttr2IntVector(const Table &resTable, std::vector<std::vector<ui>> &recordsA,
									   std::vector<std::vector<ui>> &recordsB, std::vector<double> &weightsA,
									   std::vector<double> &weightsB, std::vector<double> &wordwt,
									   ui columnA, ui columnB, TokenizerType tok_type,
									   ui &num_word, ui q)
{
	ui num_row = resTable.rows.size();

	std::unordered_map<std::string, std::vector<std::pair<ui, ui>>> inv_index;
	std::vector<std::vector<std::string>> bowsA;
	std::vector<std::vector<std::string>> bowsB;
	std::vector<std::pair<ui, std::string>> tokens;

	bowsA.resize(num_row);
	bowsB.resize(num_row);

	switch (tok_type)
	{
	case TokenizerType::Dlm:
	{
		// std::string delim = " ,-_:;.?!|()[]\\\t\n\r";
		// std::string delim = " .,\\\t\r\n";
		std::string delim = " \"\',\\\t\r\n";
		Tokenizer::updateBagDlm(resTable, bowsA, columnA, delim, 0);
		Tokenizer::updateBagDlm(resTable, bowsB, columnB, delim, 0);
		break;
	}
	case TokenizerType::QGram:
	{
		Tokenizer::updateBagQGram(resTable, bowsA, columnA, q);
		Tokenizer::updateBagQGram(resTable, bowsB, columnB, q);
		break;
	}
	case TokenizerType::WSpace:
	{
		std::string delim = " ";
		Tokenizer::updateBagDlm(resTable, bowsA, columnA, delim, 1);
		Tokenizer::updateBagDlm(resTable, bowsB, columnB, delim, 1);
		break;
	}
	case TokenizerType::AlphaNumeric:
	{
		Tokenizer::updateBagAlphaNumeric(resTable, bowsA, columnA);
		Tokenizer::updateBagAlphaNumeric(resTable, bowsB, columnB);
		break;
	}

	default:
	{
		printf("No such tokenizers\n");
		exit(1);
	}
	}

	printf("Building tokens index....\n");
	fflush(stdout);

	// Update index & tokens
	for (ui i = 0; i < num_row; i++)
	{
		for (auto &index_str : bowsA[i])
		{
			auto &str = index_str; // token
#if SKIP_NO_ALPHANUMERIC == 1
			if (str.empty() || str == " ")
				continue;
#endif
			inv_index[str].emplace_back(i, 0);
		}
	}
	for (ui i = 0; i < num_row; i++)
	{
		for (auto &index_str : bowsB[i])
		{
			auto &str = index_str; // token
#if SKIP_NO_ALPHANUMERIC == 1
			if (str.empty() || str == " ")
				continue;
#endif
			inv_index[str].emplace_back(i, 1);
		}
	}
	num_word = inv_index.size();
	for (auto &entry : inv_index)
		tokens.emplace_back(entry.second.size(), entry.first);

	// Sort according to the token's frequency
	std::sort(tokens.begin(), tokens.end(),
			  [](const std::pair<int, std::string> &p1, const std::pair<int, std::string> &p2)
			  {
				  return p1.first < p2.first;
			  });

#ifdef CHECK_TOKENIZE
	for (ui i = 0; i < tokens.size(); i++)
		all_index[i] = tokens[i].second;
#endif

	// Update records
	std::vector<std::vector<ui>> datasetsA(num_row);
	std::vector<std::vector<ui>> datasetsB(num_row);

	ui num_tokens = tokens.size();
	ui rec_num = num_row + num_row;
	for (ui i = 0; i < num_tokens; i++)
	{
		auto &word = tokens[i].second;
		for (auto &j : inv_index[word])
		{
			if (!j.second) // A
				datasetsA[j.first].emplace_back(i);
			else // B
				datasetsB[j.first].emplace_back(i);
		}

		// word weight
		ui freq = tokens[i].first;
		wordwt.emplace_back(log10(rec_num * 1.0 / freq));
	}

	// for(ui i = 0; i < num_tokens; i++) {
	// 	ui freq = tokens[i].first;
	// 	weights.emplace_back(log10(num_row * 1.0 / freq));
	// }

	for (ui i = 0; i < num_row; i++)
		recordsA.emplace_back(datasetsA[i]);
	for (ui i = 0; i < num_row; i++)
		recordsB.emplace_back(datasetsB[i]);

	// record weight
	weightsA.resize(num_row, 0.0);
	weightsB.resize(num_row, 0.0);

	for (ui i = 0; i < num_row; i++)
		for (const auto &w : recordsA[i])
			weightsA[i] += wordwt[w];
	for (ui i = 0; i < num_row; i++)
		for (const auto &w : recordsB[i])
			weightsB[i] += wordwt[w];

			// Remember to disable sort before checking
#ifdef CHECK_TOKENIZE
			// FILE* fpCheck = fopen("./buffer/check_tokens.txt", "w");
			// for(ui i = 0; i < num_row; i++) {
			// 	for(ui j = 0; j < num_col; j++) {
			// 		for(ui k = offsets[i][j]; k < offsets[i][j + 1]; k++)
			// 			fprintf(fpCheck, "%s ", all_index[records[i][k]].c_str());
			// 		fprintf(fpCheck, "\t");
			// 	}
			// 	fprintf(fpCheck, "\n");
			// }
			// fclose(fpCheck);
#endif
}