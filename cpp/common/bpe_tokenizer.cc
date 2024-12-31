/*
 * author: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#include "common/bpe_tokenizer.h"


std::vector<wchar_t> BPETokenizer::puncsChinese = {L'。', L'，', L'？', L'！', L'；', L'：', L'、', L'（', L'）', L'「',
												   L'」', L'“', L'”', L'‘', L'’', L'《', L'》', L'【', L'】', L'—', L'～',
												   L'　'};

std::wstring BPETokenizer::replacePuncs(const std::wstring &ws)
{
	std::wstring pattern = L"[";
	for (const auto &punct : puncsChinese)
	{
		pattern += punct;
	}
	pattern += L"]";

	std::wregex regexPattern(pattern);

	return std::regex_replace(ws, regexPattern, L"#");
}


// void BPETokenizer::tokenize()
// {
// 	unordered_map<string, uint32_t> vocab;
// 	if (strcmp(vocabPath, "") != 0)
// 	{
// 		readVocab(vocabPath, vocab);
// 	}

// 	// read codes
// 	unordered_map<tps, uint32_t, pair_hash> codes;
// 	unordered_map<string, tps> reversed_codes;
// 	readCodes(codesPath, codes, reversed_codes);

// 	// read input file words
// 	unordered_map<string, uint32_t> word_count;
// 	readText(inputFile, word_count);

// 	// tokenize
// 	unordered_map<string, vector<string>> bpeTok;
// 	tokenize_str(word_count, bpeTok);

// 	vector<pair<string, vector<string>>> bpeTokVec;
// 	for (auto x : bpeTok)
// 	{
// 		bpeTokVec.push_back(x);
// 	}

// 	// apply BPE codes to each word
// 	unordered_map<string, string> bpe[kThreads];
// 	vector<thread> threads;
// 	for (size_t i = 0; i < kThreads; i++)
// 	{
// 		threads.emplace_back(
// 			[&](size_t this_thread)
// 			{
// 				for (size_t w = this_thread; w < bpeTokVec.size(); w += kThreads)
// 				{
// 					auto &x = bpeTokVec[w];
// 					bpe[this_thread][x.first] = process_bpe(x.second, codes, reversed_codes, vocab);
// 				}
// 			},
// 			i);
// 	}

// 	unordered_map<string, string> final_bpe;
// 	for (size_t i = 0; i < kThreads; i++)
// 	{
// 		threads[i].join();
// 		for (auto x : bpe[i])
// 		{
// 			final_bpe[x.first] = x.second;
// 		}
// 	}
// 	// output
// 	outputText(outputFile, inputFile, final_bpe);
// }