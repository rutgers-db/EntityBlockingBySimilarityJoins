/*
 * author: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#include "feature/cal_feature.h"

FeatureIndex CalculateFeature::index;


void CalculateFeature::calOriginalFeatures(std::vector<std::vector<double>> &featureValues, const std::string &func, 
                                           const std::string &lstr, const std::string &rstr, 
                                           const std::vector<std::string> &ltokens, 
                                           const std::vector<std::string> &rtokens, 
                                           bool isCoeff)
{
    if(func == "jac" || func == "cos" || func == "dice") {
        CalculateFeature::SetJoinFunc setJoinP = nullptr;
        if(func == "jac") setJoinP = &FeatureUtils::jaccard;
        else if(func == "cos") setJoinP = &FeatureUtils::cosine;
        else if(func == "dice") setJoinP = &FeatureUtils::dice;
        featureValues.back().emplace_back(setJoinP(ltokens, rtokens));
    }
    else if(func == "overlap") {
        if(!isCoeff) featureValues.back().emplace_back((double)FeatureUtils::overlap(ltokens, rtokens));
        else featureValues.back().emplace_back(FeatureUtils::overlapCoeff(ltokens, rtokens));
    }
    else {
        CalculateFeature::StringJoinFunc stringJoinP = nullptr;
        if(func == "lev") stringJoinP = &FeatureUtils::levDist;
        else if(func == "anm") stringJoinP = &FeatureUtils::absoluteNorm;
        else if(func == "exm") stringJoinP = &FeatureUtils::exactMatch;
        featureValues.back().emplace_back(stringJoinP(lstr, rstr));
    }
}


void CalculateFeature::calOneSideFeatures(std::vector<std::vector<double>> &featureValues, SetJoinFunc setJoinP, const std::string &tok, 
                                          const std::vector<std::string> &tokens, const std::vector<std::string> &ictokens, 
                                          const FeatureIndex::GroupToken &curGrpDlm, const FeatureIndex::GroupToken &curGrpQgm, 
                                          int cltid, int iccltid)
{
    double maxVal = setJoinP(tokens, ictokens);
    const auto &docs = tok == "dlm" ? curGrpDlm.at(iccltid) : curGrpQgm.at(iccltid);
    for(const auto &doc : docs) {
        double newVal = setJoinP(doc, tokens);
        maxVal = std::max(maxVal, newVal);
    }
    featureValues.back().emplace_back(maxVal);
}


void CalculateFeature::calOneSideFeatures(std::vector<std::vector<double>> &featureValues, StringJoinFunc stringJoinP, const std::string &tok, 
                                          const std::string &str, const std::string &icstr, const FeatureIndex::Group &curGrp,
                                          int cltid, int iccltid, const std::string &func)
{
    double maxVal = stringJoinP(str, icstr);
    const auto &docs = curGrp.at(iccltid);
    for(const auto &doc : docs) {
        double newVal = stringJoinP(doc, str);
        if(func == "lev") 
            maxVal = std::min(std::abs(maxVal), std::abs(newVal));
        else 
            maxVal = std::max(maxVal, newVal);
    }
    featureValues.back().emplace_back(maxVal);
}


void CalculateFeature::calDoubleSideFeatures(std::vector<std::vector<double>> &featureValues, SetJoinFunc setjoinP, const std::string &tok, 
                                             const std::vector<std::string> &ltokens, const std::vector<std::string> &rtokens, 
                                             const FeatureIndex::GroupToken &curGrpDlm, const FeatureIndex::GroupToken &curGrpQgm, 
                                             int lcltid, int rcltid, int *const &curDCIdx, double ***const &curCache, 
                                             const std::vector<int> &featureLength, const std::string &func, ui attrpos)
{
    if(lcltid == rcltid) {
        if(func == "overlap") {
            const auto &ldocs = tok == "dlm" ? curGrpDlm.at(lcltid) : curGrpQgm.at(lcltid);
            size_t maxlen = 0;
            for(const auto &ldoc : ldocs)
                maxlen = std::max(ldoc.size(), maxlen);
            featureValues.back().emplace_back(maxlen * 1.0);
        } 
        else featureValues.back().emplace_back(1.0);
    }
    else {
        double maxVal = setjoinP(ltokens, rtokens);
        const auto &ldocs = tok == "dlm" ? curGrpDlm.at(lcltid) : curGrpQgm.at(lcltid);
        const auto &rdocs = tok == "dlm" ? curGrpDlm.at(rcltid) : curGrpQgm.at(rcltid);

        if(ldocs.size() >= index.MIN_CACHED_LENGTH && rdocs.size() >= index.MIN_CACHED_LENGTH) {
            int ldidx = curDCIdx[lcltid];
            int rdidx = curDCIdx[rcltid];
            int colid = index.calCahceIndex(func, tok, featureLength[attrpos]);
            maxVal = std::max(curCache[ldidx][rdidx][colid], maxVal);
        }
        else {
            for(const auto &ldoc : ldocs) {
                for(const auto &rdoc : rdocs) {
                    double newVal = setjoinP(ldoc, rdoc);
                    maxVal = std::max(maxVal, newVal);
                }
            }
        }

        featureValues.back().emplace_back(maxVal);
    }
}


void CalculateFeature::calDoubleSideFeatures(std::vector<std::vector<double>> &featureValues, StringJoinFunc stringJoinP, const std::string &tok, 
                                             const std::string &lstr, const std::string &rstr, const FeatureIndex::Group &curGrp, 
                                             int lcltid, int rcltid, int *const &curDCIdx, double ***const &curCache, 
                                             const std::vector<int> &featureLength, const std::string &func, ui attrpos)
{
    if(lcltid == rcltid) {
        if(func == "lev") featureValues.back().emplace_back(0.0);
        else if(func == "anm") featureValues.back().emplace_back(1.0);
        else if(func == "exm") featureValues.back().emplace_back(1.0);
    }
    else {
        double maxVal = stringJoinP(lstr, rstr);
        const auto &ldocs = curGrp.at(lcltid);
        const auto &rdocs = curGrp.at(rcltid);
        
        if(ldocs.size() >= index.MIN_CACHED_LENGTH && rdocs.size() >= index.MIN_CACHED_LENGTH) {
            int ldidx = curDCIdx[lcltid];
            int rdidx = curDCIdx[rcltid];
            int colid = index.calCahceIndex(func, tok, featureLength[attrpos]);
            if(func == "lev")
                maxVal = std::min(std::abs(curCache[ldidx][rdidx][colid]), std::abs(maxVal));
            else
                maxVal = std::max(curCache[ldidx][rdidx][colid], maxVal);
        }
        else {
            for(const auto &ldoc : ldocs) {
                for(const auto &rdoc : rdocs) {
                    double newVal = stringJoinP(ldoc, rdoc);
                    if(func == "lev") 
                        maxVal = std::min(std::abs(maxVal), std::abs(newVal));
                    else 
                        maxVal = std::max(maxVal, newVal);
                }
            }
        }

        featureValues.back().emplace_back(maxVal);
    }
}


void CalculateFeature::calAll(int numFeatures, Rule *featureNames, const std::vector<std::string> &attrVec, const Table &resTable, 
                              std::vector<std::vector<double>> &featureValues, const FeatureIndex::Groups &group, 
                              const FeatureIndex::GroupTokens &groupTokensDlm, const FeatureIndex::GroupTokens &groupTokensQgm, 
                              const FeatureIndex::Cluster &cluster, const std::vector<int> &featureLength, bool flagConsistent, 
                              bool isTopK)
{
    for(int ridx = 0; ridx < resTable.row_no; ridx ++) {
        const auto &curRow = resTable.rows[ridx];
        int lid = std::stoi(curRow[1]);
        int rid = std::stoi(curRow[2]);
        featureValues.emplace_back();

        for(int j = 0; j < numFeatures; j ++) {
            std::string attr = featureNames[j].attr;
            std::string func = featureNames[j].sim;
            std::string tok = featureNames[j].tok;
            TokenizerType tokType = tok == "dlm" ? TokenizerType::Dlm 
                                                 : TokenizerType::QGram;

            std::string lsch = "ltable_" + attr;
            std::string rsch = "rtable_" + attr;
            ui lpos = resTable.inverted_schema.at(lsch);
            ui rpos = resTable.inverted_schema.at(rsch);
            std::string lstr = curRow[lpos];
            std::string rstr = curRow[rpos];
            std::vector<std::string> ltokens;
            std::vector<std::string> rtokens;
            FeatureUtils::tokenize(lstr, tokType, ltokens);
            FeatureUtils::tokenize(rstr, tokType, rtokens);

            auto iter = std::find(attrVec.begin(), attrVec.end(), attr);
            if(iter == attrVec.end()) {
                CalculateFeature::calOriginalFeatures(featureValues, func, lstr, rstr, ltokens, rtokens, isTopK);
                continue;
            }
            
            ui attrpos = std::distance(attrVec.begin(), iter);
            const auto &curGrp = group[attrpos];
            const auto &curGrpDlm = groupTokensDlm[attrpos];
            const auto &curGrpQgm = groupTokensQgm[attrpos];
            const auto &curClt = cluster[attrpos];
            bool lhaskey = curClt.find(lstr) != curClt.end();
            bool rhaskey = curClt.find(rstr) != curClt.end();
            int lcltid = lhaskey ? curClt.at(lstr) : -1;
            int rcltid = rhaskey ? curClt.at(rstr) : -1;
            // cache & index
            const auto &curCache = index.featureValCache[attrpos];
            const auto &curDCIdx = index.discreteCacheIdx[attrpos];

            if(flagConsistent) {
                if(lhaskey) {
                    const auto &docs = curGrp.at(lcltid);
                    const auto &tokDocs = tok == "dlm" ? curGrpDlm.at(lcltid) : curGrpQgm.at(lcltid);
                    lstr = docs[0];
                    ltokens = tokDocs[0];
                }
                if(rhaskey) {
                    const auto &docs = curGrp.at(rcltid);
                    const auto &tokDocs = tok == "dlm" ? curGrpDlm.at(rcltid) : curGrpQgm.at(rcltid);
                    rstr = docs[0];
                    rtokens = tokDocs[0];
                }

                CalculateFeature::calOriginalFeatures(featureValues, func, lstr, rstr, ltokens, rtokens, isTopK);
                continue;
            }

            if(func == "jac" || func == "cos" || func == "dice" || func == "overlap") {
                CalculateFeature::SetJoinFunc setJoinP = nullptr;
                if(func == "jac") 
                    setJoinP = &FeatureUtils::jaccard;
                else if(func == "cos") 
                    setJoinP = &FeatureUtils::cosine;
                else if(func == "dice") 
                    setJoinP = &FeatureUtils::dice;
                else if(func == "overlap") {
                    if(isTopK) setJoinP = &FeatureUtils::overlapCoeff;
                    else setJoinP = &FeatureUtils::overlapD;
                }
                
                if(!lhaskey && !rhaskey)
                    CalculateFeature::calOriginalFeatures(featureValues, func, lstr, rstr, ltokens, rtokens, isTopK);
                else if(lhaskey && !rhaskey)
                    CalculateFeature::calOneSideFeatures(featureValues, setJoinP, tok, rtokens, ltokens, 
                                                          curGrpDlm, curGrpQgm, rcltid, lcltid);
                else if(!lhaskey && rhaskey) 
                    CalculateFeature::calOneSideFeatures(featureValues, setJoinP, tok, ltokens, rtokens, 
                                                          curGrpDlm, curGrpQgm, lcltid, rcltid);
                else
                    CalculateFeature::calDoubleSideFeatures(featureValues, setJoinP, tok, ltokens, rtokens, 
                                                             curGrpDlm, curGrpQgm, lcltid, rcltid, curDCIdx, 
                                                             curCache, featureLength, func, attrpos);
            }
            // lev, anm, exm
            else {
                CalculateFeature::StringJoinFunc stringJoinP = nullptr;
                if(func == "lev") 
                    stringJoinP = &FeatureUtils::levDist;
                else if(func == "anm") 
                    stringJoinP = &FeatureUtils::absoluteNorm;
                else if(func == "exm") 
                    stringJoinP = &FeatureUtils::exactMatch;

                if(!lhaskey && !rhaskey)
                    CalculateFeature::calOriginalFeatures(featureValues, func, lstr, rstr, ltokens, rtokens, isTopK);
                else if(lhaskey && !rhaskey) 
                    CalculateFeature::calOneSideFeatures(featureValues, stringJoinP, tok, rstr, lstr, curGrp, 
                                                          rcltid, lcltid, func);
                else if(!lhaskey && rhaskey) 
                    CalculateFeature::calOneSideFeatures(featureValues, stringJoinP, tok, lstr, rstr, curGrp, 
                                                          lcltid, rcltid, func);
                else if(lhaskey && rhaskey) 
                    CalculateFeature::calDoubleSideFeatures(featureValues, stringJoinP, tok, lstr, rstr, curGrp, 
                                                             lcltid, rcltid, curDCIdx, curCache, featureLength, 
                                                             func, attrpos);
            }
        }
    }
}


void CalculateFeature::calAllWithoutInterchangeable(int numFeatures, Rule *featureNames, const std::vector<std::string> &attrVec, const Table &resTable, 
                                                    std::vector<std::vector<double>> &featureValues, const std::vector<int> &featureLength, bool isTopK)
{
    for(int ridx = 0; ridx < resTable.row_no; ridx ++) {
        const auto &curRow = resTable.rows[ridx];
        int lid = std::stoi(curRow[1]);
        int rid = std::stoi(curRow[2]);
        featureValues.emplace_back();

        for(int j = 0; j < numFeatures; j ++) {
            std::string attr = featureNames[j].attr;
            std::string func = featureNames[j].sim;
            std::string tok = featureNames[j].tok;
            TokenizerType tokType = tok == "dlm" ? TokenizerType::Dlm 
                                                 : TokenizerType::QGram;

            std::string lsch = "ltable_" + attr;
            std::string rsch = "rtable_" + attr;
            ui lpos = resTable.inverted_schema.at(lsch);
            ui rpos = resTable.inverted_schema.at(rsch);
            std::string lstr = curRow[lpos];
            std::string rstr = curRow[rpos];
            std::vector<std::string> ltokens;
            std::vector<std::string> rtokens;
            FeatureUtils::tokenize(lstr, tokType, ltokens);
            FeatureUtils::tokenize(rstr, tokType, rtokens);

            if(func == "jac" || func == "cos" || func == "dice" || func == "overlap") {
                /*
                CalculateFeature::SetJoinFunc setJoinP = nullptr;
                if(func == "jac") 
                    setJoinP = (double (*)(const std::vector<std::string> &, const std::vector<std::string> &))FeatureUtils::jaccard;
                else if(func == "cos") 
                    setJoinP = (double (*)(const std::vector<std::string> &, const std::vector<std::string> &))FeatureUtils::cosine;
                else if(func == "dice") 
                    setJoinP = (double (*)(const std::vector<std::string> &, const std::vector<std::string> &))FeatureUtils::dice;
                else if(func == "overlap") 
                    setJoinP = isTopK ? (double (*)(const std::vector<std::string> &, const std::vector<std::string> &))FeatureUtils::overlapCoeff 
                                      : (double (*)(const std::vector<std::string> &, const std::vector<std::string> &))FeatureUtils::overlapD;
                */

                CalculateFeature::calOriginalFeatures(featureValues, func, lstr, rstr, ltokens, rtokens, isTopK);
            }
            // lev, anm, exm
            else {
                /*
                CalculateFeature::StringJoinFunc stringJoinP = nullptr;
                if(func == "lev") 
                    stringJoinP = (double (*)(const std::string &, const std::string &))FeatureUtils::levDist;
                else if(func == "anm") 
                    stringJoinP = (double (*)(const std::string &, const std::string &))FeatureUtils::absoluteNorm;
                else if(func == "exm") 
                    stringJoinP = (double (*)(const std::string &, const std::string &))FeatureUtils::exactMatch;
                */

                CalculateFeature::calOriginalFeatures(featureValues, func, lstr, rstr, ltokens, rtokens, isTopK);
            }
        }
    }
}