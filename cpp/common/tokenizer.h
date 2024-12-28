/*
 * author: Chaoji Zuo and Zhizhi Wang in rutgers-db/SIGMOD2022-Programming-Contest-Public
 * modified: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#ifndef _TOKENIZER_H_
#define _TOKENIZER_H_

#include "common/config.h"
#include "common/type.h"
#include "common/dataframe.h"
#include <regex>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <ctype.h>
#include <codecvt>

// If we need to do q-gram tokenization, we could use text normalization.
// That is only keep a-z/A-Z and 0-9 in string.
// If we need to use dlm tokenization, we should not use text normalization. 
// Since some tokens like "18.9" may be eliminated in this way.


class Tokenizer
{
public:
	Tokenizer() = default;
	~Tokenizer() = default;
	Tokenizer(const Tokenizer& other) = delete;
	Tokenizer(Tokenizer&& other) = delete;

private:
	// check a string if contains other chars
	static bool isNotAlphaNumeric(char c);
	static bool isAlphaNumeric(const std::string &s);

public:
	/*
	 * As suggested in sparkly, it's better to tokenize strings only keeping alphanumeric chars.
	 * For dlm_dc0 & q(3)-gram, they will introduce non-alphanumeric characters
	 * We could set marco "SKIP_NO_ALPHANUMERIC" to avoid this
	 * Alternatively, we could also use alphanumeric tokenizer
	 * And we already synethsis the difference between py_entitymatching & blocker
	 */
	// dlm_dc0: tokenize by white space, "\n" or "\t", etc.
	static void string2TokensDlm(const std::string &s, std::vector<std::string> &res, 
							  	 const std::string &delims);
	// q-gram: default 3-qgram with padding
	static void string2TokensQGram(const std::string &s, std::vector<std::string> &res, 
								   ui q);
	// w-space: it's the same function as dlm_dc0
	static void string2TokensWSpace(const std::string &s, std::vector<std::string> &res);
	// alphanumeric: returns a list of tokens that are maximal sequences of consecutive alphanumeric characters.
	static void string2TokensAlphaNumeric(const std::string &s, std::vector<std::string> &res);

public:	
	// Convert a table into a set of int vectors
	static void stringNormalize(std::string& s, ui startegy);
	static void updateBagDlm(const Table& table, std::vector<std::vector<std::string>>& bow, 
							 ui column, const std::string& dlim, ui strategy);
	static void updateBagQGram(const Table& table, std::vector<std::vector<std::string>>& bow, 
							   ui column, ui q);
	static void updateBagAlphaNumeric(const Table &table, std::vector<std::vector<std::string>>& bow, 
									  ui column);
	static void sortIdMap(std::vector<ui>& id_map, const std::vector<std::vector<ui>>& datasets);

public:
	// weightsA & weightsB are same
	// they record the word weight
	// which is log_10(records_num / word_frequency)
	static void RStableAttr2IntVector(const Table& tableA, const Table& tableB, 
									  std::vector<std::vector<ui>>& recordsA, 
									  std::vector<std::vector<ui>>& recordsB, 
									  std::vector<double>& weightsA, 
									  std::vector<double>& weightsB, 
									  std::vector<double>& wordwt,
									  std::vector<ui>& id_mapA,
									  std::vector<ui>& id_mapB, 
									  ui columnA, ui columnB, 
									  TokenizerType tok_type, ui& num_word, ui q);
	static void SelftableAttr2IntVector(const Table& tableA, 
									    std::vector<std::vector<ui>>& recordsA, 
									  	std::vector<double>& weightsA, 
										std::vector<double>& wordwt,
									  	std::vector<ui>& id_mapA,
									  	ui columnA, TokenizerType tok_type, 
										ui& num_word, ui q);
	// tokenize the sample / match res table
	static void resTableAttr2IntVector(const Table &resTable, std::vector<std::vector<ui>> &recordsA, 
									   std::vector<std::vector<ui>> &recordsB, std::vector<double> &weightsA,
									   std::vector<double> &weightsB, std::vector<double> &wordwt, 
									   ui columnA, ui columnB, TokenizerType tok_type, 
									   ui &num_word, ui q);
};

#endif // _TOKENIZER_H_