/*
 * author: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#include "feature/feature_index.h"


// get the column index for a specific feature
int FeatureIndex::calCahceIndex(const std::string &func, const std::string &tok, int numFeature)
{
    int idx = 0;

    if(func == "exm" || func == "lev") {
        idx = func == "lev" ? 4 : 5;
        return idx;
    }

    if(tok == "dlm") idx = 0;
    else if(tok == "qgm") idx = numFeature == 6 ? 0 : 1;

    if(func == "jac") idx = idx * 4 + 0;
    else if(func == "cos") idx = idx * 4 + 1;
    else if(func == "dice") idx = idx * 4 + 2;
    else if(func == "overlap") idx = idx * 4 + 3;

    return idx;
}


void FeatureIndex::releaseMemory()
{
    int attrSize = (int)attrCahceLength.size();
    for(int i = 0; i < attrSize; i++) {
        for(int j = 0; j < attrCahceLength[i]; j++) {
            for(int k = 0; k < attrCahceLength[i]; k++)
                delete[] featureValCache[i][j][k];
            delete[] featureValCache[i][j];
        }
        delete[] featureValCache[i];
        delete[] discreteCacheIdx[i];
    }
    delete[] featureValCache;
    delete[] discreteCacheIdx;

    featureValCache = nullptr;
    discreteCacheIdx = nullptr;
    attrCahceLength.clear();
}


// sort according to idf
void FeatureIndex::normalizeTokens(const GroupTokens &grpToks, GroupTokensInt &grpToksInt)
{
    grpToksInt.resize(grpToks.size());

    struct IndexEntity {
        int keyId{0}; // key in group
        int posId{0}; // pos in vector
        IndexEntity() = default;
        IndexEntity(int _keyId, int _posId) : keyId(_keyId), posId(_posId) { }
    };

    int totalAttr = (int)grpToks.size();
    for(int i = 0; i < totalAttr; i++) {
        const auto &curGrpToks = grpToks[i];
        auto &curGrpToksInt = grpToksInt[i];
        std::unordered_map<std::string, std::vector<IndexEntity>> inv_index;

        for(const auto &it : curGrpToks) {
            int keyId = it.first;
            const auto &vec = it.second;
            int size = (int)vec.size();
            for(int j = 0; j < size; j++) {
                for(const auto &str : vec[j])
                    inv_index[str].emplace_back(keyId, j);
            }

            curGrpToksInt.emplace(keyId, std::vector<std::vector<ui>> (size));
        }

        std::vector<std::pair<int, std::string>> tokens;
        for(auto& entry : inv_index)
		    tokens.emplace_back((int)entry.second.size(), entry.first);
        std::sort(tokens.begin(), tokens.end(), 
                [](const std::pair<int, std::string> &p1, const std::pair<int, std::string> &p2){
                    return p1.first < p2.first;
        });

        ui num_tokens = tokens.size();
        for(ui j = 0; j < num_tokens; j++) {
            const auto &word = tokens[j].second;
            for(const auto &l : inv_index[word]) {
                int keyId = l.keyId;
                int posId = l.posId;
                auto &toks = curGrpToksInt.at(keyId);
                toks[posId].emplace_back(j);
            }
        }

        for(auto &iter : curGrpToksInt) {
            std::sort(iter.second.begin(), iter.second.end(), [](const std::vector<ui> &lhs, const std::vector<ui> &rhs) {
                return lhs.size() < rhs.size();
            });
            if(iter.second.back().empty()) {
                std::cerr << "wrong size" << std::endl;
                std::cerr << inv_index.size() << std::endl;
                exit(1);
            }
        }
    }
}


void FeatureIndex::calIndexLength4(int curAttrIdx, const std::vector<int> &grpid, const GroupToken &curGrpDlm, const GroupTokenInt &curGrpDlmInt, 
                                   bool isCoeff)
{
    int bucketSize = (int)grpid.size();

#pragma omp parallel for
    for(int id1 = 0; id1 < bucketSize; id1++) {
        int lid = grpid[id1];
        const auto &ldlms = curGrpDlm.at(lid);
        const auto &lDlmsInt = curGrpDlmInt.at(lid);
        for(int id2 = id1 + 1; id2 < bucketSize; id2++) {
            int rid = grpid[id2];
            const auto &rdlms = curGrpDlm.at(rid);
            const auto &rDlmsInt = curGrpDlmInt.at(rid);

            double maxJacVal = 0.0;
            double maxCosVal = 0.0;
            double maxDiceVal = 0.0;
            double maxOvlpVal = 0.0;
            for(const auto &ldoc : ldlms) {
                int lsize = (int)ldoc.size();
                for(const auto &rdoc : rdlms) {
                    int rsize = (int)rdoc.size();
                    if(rsize < lsize * diceLengthFilter) continue;
                    if(rsize > lsize / diceLengthFilter) break;

                    int ovlpVal = FeatureUtils::overlap(ldoc, rdoc);
                    if(std::abs(FeatureUtils::NaN - ovlpVal) < 1e-5) {
                        std::cerr << "error in fast group returns NaN" << std::endl;
                        std::cerr << lid << " " << lsize << " " << rid << " " << rsize << std::endl;
                        exit(1);
                    }

                    if(!isCoeff) maxOvlpVal = std::max(maxOvlpVal, ovlpVal * 1.0);
                    else maxOvlpVal = std::max(maxOvlpVal, FeatureUtils::overlapCoeff(ldoc, rdoc, ovlpVal));

                    // length filter
                    if(rsize >= lsize * universalDet && rsize <= lsize / universalDet)
                        maxJacVal = std::max(maxJacVal, FeatureUtils::jaccard(ldoc, rdoc, ovlpVal));
                    if(rsize >= lsize * cosLengthFilter && rsize <= lsize / cosLengthFilter)
                        maxCosVal = std::max(maxCosVal, FeatureUtils::cosine(ldoc, rdoc, ovlpVal));
                    if(rsize >= lsize * diceLengthFilter && rsize <= lsize / diceLengthFilter)
                        maxDiceVal = std::max(maxDiceVal, FeatureUtils::dice(ldoc, rdoc, ovlpVal));
                }
            }

            featureValCache[curAttrIdx][id1][id2][0] = maxJacVal;
            featureValCache[curAttrIdx][id2][id1][0] = maxJacVal;
            featureValCache[curAttrIdx][id1][id2][1] = maxCosVal;
            featureValCache[curAttrIdx][id2][id1][1] = maxCosVal;
            featureValCache[curAttrIdx][id1][id2][2] = maxDiceVal;
            featureValCache[curAttrIdx][id2][id1][2] = maxDiceVal;
            featureValCache[curAttrIdx][id1][id2][3] = maxOvlpVal;
            featureValCache[curAttrIdx][id2][id1][3] = maxOvlpVal;
        }
    }
}


void FeatureIndex::calIndexLength6(int curAttrIdx, const std::vector<int> &grpid, const Group &curGrp, const GroupToken &curGrpQgm, 
                                   const GroupTokenInt &curGrpQgmInt, bool isCoeff)
{
    int bucketSize = (int)grpid.size();

#pragma omp parallel for
    for(int id1 = 0; id1 < bucketSize; id1++) {
        int lid = grpid[id1];
        const auto &ldocs = curGrp.at(lid);
        const auto &lqgms = curGrpQgm.at(lid);
        for(int id2 = id1 + 1; id2 < bucketSize; id2++) {
            int rid = grpid[id2];
            const auto &rdocs = curGrp.at(rid);
            const auto &rqgms = curGrpQgm.at(rid);

            double maxJacVal = 0.0;
            double maxCosVal = 0.0;
            double maxDiceVal = 0.0;
            double maxOvlpVal = 0;
            double minLevVal = 100000.0;
            double maxExmVal = 0.0;

            for(const auto &ldoc : lqgms) {
                for(const auto &rdoc : rqgms) {
                    int ovlpVal = FeatureUtils::overlap(ldoc, rdoc);
                    if(std::abs(FeatureUtils::NaN - ovlpVal) < 1e-5) {
                        std::cerr << "error in fast group returns NaN" << std::endl;
                        std::cerr << lid << " " << ldoc.size() << " " << rid << " " << rdoc.size() << std::endl;
                        exit(1);
                    }

                    if(!isCoeff) maxOvlpVal = std::max(maxOvlpVal, ovlpVal * 1.0);
                    else maxOvlpVal = std::max(maxOvlpVal, FeatureUtils::overlapCoeff(ldoc, rdoc, ovlpVal));

                    maxJacVal = std::max(maxJacVal, FeatureUtils::jaccard(ldoc, rdoc, ovlpVal));
                    maxCosVal = std::max(maxCosVal, FeatureUtils::cosine(ldoc, rdoc, ovlpVal));
                    maxDiceVal = std::max(maxDiceVal, FeatureUtils::dice(ldoc, rdoc, ovlpVal));
                }
            }

            for(const auto &ldoc : ldocs) {
                for(const auto &rdoc : rdocs) {
                    minLevVal = std::min(minLevVal, FeatureUtils::levDist(ldoc, rdoc));
                    maxExmVal = std::max(maxExmVal, FeatureUtils::exactMatch(ldoc, rdoc));
                }
            }

            featureValCache[curAttrIdx][id1][id2][0] = maxJacVal;
            featureValCache[curAttrIdx][id2][id1][0] = maxJacVal;
            featureValCache[curAttrIdx][id1][id2][1] = maxCosVal;
            featureValCache[curAttrIdx][id2][id1][1] = maxCosVal;
            featureValCache[curAttrIdx][id1][id2][2] = maxDiceVal;
            featureValCache[curAttrIdx][id2][id1][2] = maxDiceVal;
            featureValCache[curAttrIdx][id1][id2][3] = maxOvlpVal;
            featureValCache[curAttrIdx][id2][id1][3] = maxOvlpVal;
            featureValCache[curAttrIdx][id1][id2][4] = minLevVal;
            featureValCache[curAttrIdx][id2][id1][4] = minLevVal;
            featureValCache[curAttrIdx][id1][id2][5] = maxExmVal;
            featureValCache[curAttrIdx][id2][id1][5] = maxExmVal;
        }
    }
}


void FeatureIndex::calIndexLength8(int curAttrIdx, const std::vector<int> &grpid, const GroupToken &curGrpDlm, const GroupTokenInt &curGrpDlmInt, 
                                   const GroupToken &curGrpQgm, const GroupTokenInt &curGrpQgmInt, bool isCoeff)
{
    int bucketSize = (int)grpid.size();

#pragma omp parallel for
    for(int id1 = 0; id1 < bucketSize; id1++) {
        int lid = grpid[id1];
        const auto &ldlms = curGrpDlm.at(lid);
        const auto &lqgms = curGrpQgm.at(lid);
        for(int id2 = id1 + 1; id2 < bucketSize; id2++) {
            int rid = grpid[id2];
            const auto &rdlms = curGrpDlm.at(rid);
            const auto &rqgms = curGrpQgm.at(rid);

            double maxJacDlmVal = 0.0, maxJacQgmVal = 0.0;
            double maxCosDlmVal = 0.0, maxCosQgmVal = 0.0;
            double maxDiceDlmVal = 0.0, maxDiceQgmVal = 0.0;
            double maxOvlpDlmVal = 0.0, maxOvlpQgmVal = 0.0;

            for(const auto &ldoc : ldlms) {
                int lsize = (int)ldoc.size();
                for(const auto &rdoc : rdlms) {
                    int rsize = (int)rdoc.size();
                    if(rsize < lsize * diceLengthFilter) continue;
                    if(rsize > lsize / diceLengthFilter) break;

                    int ovlpVal = FeatureUtils::overlap(ldoc, rdoc);
                    if(std::abs(FeatureUtils::NaN - ovlpVal) < 1e-5) {
                        std::cerr << "error in fast group returns NaN" << std::endl;
                        std::cerr << lid << " " << lsize << " " << rid << " " << rsize << std::endl;
                        exit(1);
                    }

                    if(!isCoeff) maxOvlpDlmVal = std::max(maxOvlpDlmVal, ovlpVal * 1.0);
                    else maxOvlpDlmVal = std::max(maxOvlpDlmVal, FeatureUtils::overlapCoeff(ldoc, rdoc, ovlpVal));
                    
                    if(rsize >= lsize * universalDet && rsize <= lsize / universalDet)
                        maxJacDlmVal = std::max(maxJacDlmVal, FeatureUtils::jaccard(ldoc, rdoc, ovlpVal));
                    if(rsize >= lsize * cosLengthFilter && rsize <= lsize / cosLengthFilter)
                        maxCosDlmVal = std::max(maxCosDlmVal, FeatureUtils::cosine(ldoc, rdoc, ovlpVal));
                    if(rsize >= lsize * diceLengthFilter && rsize <= lsize / diceLengthFilter)
                        maxDiceDlmVal = std::max(maxDiceDlmVal, FeatureUtils::dice(ldoc, rdoc, ovlpVal));
                }
            }

            for(const auto &ldoc : lqgms) {
                int lsize = (int)ldoc.size();
                for(const auto &rdoc : rqgms) {
                    int rsize = (int)rdoc.size();
                    if(rsize < lsize * diceLengthFilter) continue;
                    if(rsize > lsize / diceLengthFilter) break;

                    int ovlpVal = FeatureUtils::overlap(ldoc, rdoc);
                    if(std::abs(FeatureUtils::NaN - ovlpVal) < 1e-5) {
                        std::cerr << "error in fast group returns NaN" << std::endl;
                        std::cerr << lid << " " << lsize << " " << rid << " " << rsize << std::endl;
                        exit(1);
                    }

                    if(!isCoeff) maxOvlpQgmVal = std::max(maxOvlpQgmVal, ovlpVal * 1.0);
                    else maxOvlpQgmVal = std::max(maxOvlpQgmVal, FeatureUtils::overlapCoeff(ldoc, rdoc, ovlpVal));

                    if(rsize >= lsize * universalDet && rsize <= lsize / universalDet)
                        maxJacQgmVal = std::max(maxJacQgmVal, FeatureUtils::jaccard(ldoc, rdoc, ovlpVal));
                    if(rsize >= lsize * cosLengthFilter && rsize <= lsize / cosLengthFilter)
                        maxCosQgmVal = std::max(maxCosQgmVal, FeatureUtils::cosine(ldoc, rdoc, ovlpVal));
                    if(rsize >= lsize * diceLengthFilter && rsize <= lsize / diceLengthFilter)
                        maxDiceQgmVal = std::max(maxDiceQgmVal, FeatureUtils::dice(ldoc, rdoc, ovlpVal));
                }
            }

            featureValCache[curAttrIdx][id1][id2][0] = maxJacDlmVal;
            featureValCache[curAttrIdx][id2][id1][0] = maxJacDlmVal;
            featureValCache[curAttrIdx][id1][id2][1] = maxCosDlmVal;
            featureValCache[curAttrIdx][id2][id1][1] = maxCosDlmVal;
            featureValCache[curAttrIdx][id1][id2][2] = maxDiceDlmVal;
            featureValCache[curAttrIdx][id2][id1][2] = maxDiceDlmVal;
            featureValCache[curAttrIdx][id1][id2][3] = maxOvlpDlmVal;
            featureValCache[curAttrIdx][id2][id1][3] = maxOvlpDlmVal;
            featureValCache[curAttrIdx][id1][id2][4] = maxJacQgmVal;
            featureValCache[curAttrIdx][id2][id1][4] = maxJacQgmVal;
            featureValCache[curAttrIdx][id1][id2][5] = maxCosQgmVal;
            featureValCache[curAttrIdx][id2][id1][5] = maxCosQgmVal;
            featureValCache[curAttrIdx][id1][id2][6] = maxDiceQgmVal;
            featureValCache[curAttrIdx][id2][id1][6] = maxDiceQgmVal;
            featureValCache[curAttrIdx][id1][id2][7] = maxOvlpQgmVal;
            featureValCache[curAttrIdx][id2][id1][7] = maxOvlpQgmVal;
        }
    }
}


// public apis:
// get number of features according to attr
int FeatureIndex::calNumFeature(const std::string attr) 
{
    if(std::count(str_gt_10w.begin(), str_gt_10w.end(), attr) != 0 
       || std::count(str_bt_5w_10w.begin(), str_bt_5w_10w.end(), attr) != 0)
        return 4;
    else if(std::count(str_eq_1w.begin(), str_eq_1w.end(), attr) != 0)
        return 6;
    else if(std::count(str_bt_1w_5w.begin(), str_bt_1w_5w.end(), attr) != 0)
        return 8;

    std::cerr << "attribute: " << attr << " is not included" << std::endl;
    return 0;
}


void FeatureIndex::globalInit(const std::vector<int> &keyNum, const std::vector<std::string> &attrs, Groups &groups, const GroupTokens &grpdlm, 
                              const GroupTokens &grpqgm, bool isCoeff)
{
    int totalAttr = (int)groups.size();
    featureValCache = new double***[totalAttr];
    discreteCacheIdx = new int*[totalAttr];

    GroupTokensInt grpDlmInt, grpQgmInt;
    normalizeTokens(grpdlm, grpDlmInt);
    normalizeTokens(grpqgm, grpQgmInt);

    for(int i = 0; i < totalAttr; i++) {
        const auto &curGrp = groups[i];
        const auto &curGrpDlm = grpdlm[i];
        const auto &curGrpQgm = grpqgm[i];
        const auto &curGrpDlmInt = grpDlmInt[i];
        const auto &curGrpQgmInt = grpQgmInt[i];
        int numFeature = calNumFeature(attrs[i]);
        if(numFeature == 0) {
            // std::cerr << "no such attr: " << attrs[i] << std::endl;
            exit(1);
        }

        std::vector<int> grpid;
        for(const auto &grpit : curGrp) {
            int keyId = grpit.first;
            if(grpit.second.size() >= MIN_CACHED_LENGTH)
                grpid.emplace_back(keyId);
        }

        // assign
        int bucketSize = (int)grpid.size();
        featureValCache[i] = new double**[bucketSize];
        for(int j = 0; j < bucketSize; j++) {
            featureValCache[i][j] = new double*[bucketSize];
            for(int l = 0; l < bucketSize; l++) {
                featureValCache[i][j][l] = new double[numFeature];
                std::fill(featureValCache[i][j][l], featureValCache[i][j][l] + numFeature, 0.0);
            }
        }

        discreteCacheIdx[i] = new int[keyNum[i]];
        std::fill(discreteCacheIdx[i], discreteCacheIdx[i] + keyNum[i], 0);
        for(int didx = 0; didx < bucketSize; didx++)
            discreteCacheIdx[i][grpid[didx]] = didx;
        
        attrCahceLength.emplace_back(bucketSize);

        // calculate
        switch(numFeature) {
            case 4 : calIndexLength4(i, grpid, curGrpDlm, curGrpDlmInt, isCoeff); break;
            case 6 : calIndexLength6(i, grpid, curGrp, curGrpQgm, curGrpQgmInt, isCoeff); break;
            case 8 : calIndexLength8(i, grpid, curGrpDlm, curGrpDlmInt, curGrpQgm, curGrpQgmInt, isCoeff); break;
        }
    }
}