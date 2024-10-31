/*
 * author: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#ifndef _KNN_BLOCKER_H_
#define _KNN_BLOCKER_H_

#include "common/tokenizer.h"
#include "common/dataframe.h"
#include "common/simfunc.h"
#include "blocker/blocker_config.h"


/*
 * k-nearest neighbors blocker
 * sim functions: cos / TF-IDF
 * tokenizers: dlm / qgm
 */
class KNNBlocker
{
public:
    KNNBlocker() = default;
    ~KNNBlocker() = default;
    KNNBlocker(const KNNBlocker &other) = delete;
    KNNBlocker(KNNBlocker &&other) = delete;

public:
    // TODO
};


#endif // _KNN_BLOCKER_H_