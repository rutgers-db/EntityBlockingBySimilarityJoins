/*
 * author: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#ifndef _TYPE_H_
#define _TYPE_H_

typedef unsigned int ui;

inline char OSseparator()
{
#ifdef _WIN32
    return '\\';
#else
    return '/';
#endif
}

/*
 * Table type: gold record or data table
 */
enum class TableType
{
	Gold = 0,
	Data = 1, 
	Invalid = 2
};


/*
 * ChunkerArrayType
 */


/*
 * Similarity functions & tokenizers
 */
enum class TokenizerType
{
	Dlm = 0,
	QGram = 1,
	WSpace = 2,
	AlphaNumeric = 3
};


enum class SimFuncType
{
	JACCARD = 0,
	COSINE = 1, 
	DICE = 2
};


/*
 * Main function
 */
enum class JoinType
{
	SELF = 0,
	RS = 1
};

enum class JoinSettings
{
	SEQUENTIAL = 0,
	PARALLEL = 1
};


#endif // _TYPE_H_