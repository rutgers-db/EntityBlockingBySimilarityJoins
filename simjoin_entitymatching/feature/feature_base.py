import py_entitymatching as em
import py_entitymatching.feature.autofeaturegen as autog
import py_entitymatching.feature.attributeutils as au
import py_entitymatching.feature.simfunctions as sim
import py_entitymatching.feature.tokenizers as tok
import pandas as pd
import numpy as np

import logging

import multiprocessing
import os

import cloudpickle
from joblib import Parallel
from joblib import delayed

import py_entitymatching.catalog.catalog_manager as cm
import py_entitymatching.utils.catalog_helper as ch
import py_entitymatching.utils.generic_helper as gh
from py_entitymatching.io.pickles import save_object, load_object
from py_entitymatching.utils.validation_helper import (
    validate_object_type,
    validate_subclass
)

logger = logging.getLogger(__name__)


sim_function_names = ['lev_dist',
                      'overlap',
                      'jaccard', 'dice',
                      'cosine',
                      'exact_match', 'abs_norm']


class NewFeatures:
    '''
    The py_entitymatching package cannot select certain features to generate.
    But our join algorithm only supports 7 sim funcs.
    So we write functions to generate supported features.
    '''

    def __init__(self):
        pass

    
    @staticmethod
    def get_supported_sim_funs():
        '''
        This function returns all the similarity functions supported by SOTA join algorithms.
        '''

        # Get all the functions
        functions = [# affine,
                # hamming_dist, hamming_sim,
                sim.lev_dist, # sim.lev_sim,
                # jaro,
                # jaro_winkler,
                # needleman_wunsch,
                # smith_waterman,
                sim.overlap, sim.jaccard, sim.dice,
                sim.cosine, # monge_elkan, 
                sim.exact_match, sim.abs_norm] # rel_diff]
        # Return a dictionary with the functions names as the key and the actual
        # functions as values.
        return dict(zip(sim_function_names, functions))
    

    @staticmethod
    def get_supported_features_for_blocking(ltable, rtable, validate_inferred_attr_types=True):
        # Validate input parameters
        # # We expect the ltable to be of type pandas DataFrame
        validate_object_type(ltable, pd.DataFrame, 'Input table A')

        # # We expect the rtable to be of type pandas DataFrame
        validate_object_type(rtable, pd.DataFrame, 'Input table B')

        # # We expect the validate_inferred_attr_types to be of type boolean
        validate_object_type(validate_inferred_attr_types, bool, 'Validate inferred attribute type')

        # Get the similarity functions to be used for blocking
        sim_funcs = NewFeatures.get_supported_sim_funs()
        # Get the tokenizers to be used for blocking
        tok_funcs = tok.get_tokenizers_for_blocking(q=[2,3,4], 
                                                    dlm_char=[[' ', '.', ',', '\\', '\t', '\r', '\n']])

        # Get the attr. types for ltable and rtable
        attr_types_ltable = au.get_attr_types(ltable)
        attr_types_rtable = au.get_attr_types(rtable)
        aindex = list(ltable)
        bindex = list(rtable)

        # Fix the attr type.
        # This is essential for generating features
        # Feature generated according to appendix in "Sigmod2017-Falcon"
        type_length = len(attr_types_ltable) - 1
        for i in range(1, type_length):
            # not consistent in l and r
            if attr_types_ltable[aindex[i]] < attr_types_rtable[bindex[i]]:
                print("\033[31mFix:\033[0m", attr_types_rtable[bindex[i]], "to", attr_types_ltable[aindex[i]], "for", aindex[i])
                attr_types_rtable[bindex[i]] = attr_types_ltable[aindex[i]]
            elif attr_types_ltable[aindex[i]] > attr_types_rtable[bindex[i]]:
                print("\033[31mFix:\033[0m", attr_types_ltable[aindex[i]], "to", attr_types_rtable[bindex[i]], "for", aindex[i])
                attr_types_ltable[aindex[i]] = attr_types_rtable[bindex[i]]
            # some specific attribute
            if aindex[i] == 'brand' or aindex[i] == 'category':
                attr_types_ltable[aindex[i]] = 'str_eq_1w'
                attr_types_rtable[bindex[i]] = 'str_eq_1w'
            elif aindex[i] == 'description':
                attr_types_ltable[aindex[i]] = 'str_gt_10w'
                attr_types_rtable[bindex[i]] = 'str_gt_10w'

        # Get the attr. correspondences between ltable and rtable
        attr_corres = au.get_attr_corres(ltable, rtable)
        # print(attr_types_ltable)
        # print(attr_corres)
        
        # Show the user inferred attribute types and features and request
        # user permission to proceed
        if validate_inferred_attr_types:
            # if the user does not want to proceed, then exit the function
            if autog.validate_attr_types(attr_types_ltable, attr_types_rtable, attr_corres) is None:
                return
        
        # Get features based on attr types, attr correspondences, sim functions
        # and tok. functions
        feature_table = autog.get_features(ltable, rtable, attr_types_ltable,
                                           attr_types_rtable, attr_corres,
                                           tok_funcs, sim_funcs)

        # Export important variables to global name space
        em._block_t = tok_funcs
        em._block_s = sim_funcs
        em._atypes1 = attr_types_ltable
        em._atypes2 = attr_types_rtable
        em._block_c = attr_corres
        # Return the feature table
        return feature_table


    @staticmethod
    def get_supported_features_for_matching(ltable, rtable, validate_inferred_attr_types=True, 
                                            at_ltable=None, at_rtable=None, dataname=None):
        # Validate input parameters
        # # We expect the ltable to be of type pandas DataFrame
        validate_object_type(ltable, pd.DataFrame, 'Input table A')

        # # We expect the rtable to be of type pandas DataFrame
        validate_object_type(rtable, pd.DataFrame, 'Input table B')

        # # We expect the validate_inferred_attr_types to be of type boolean
        validate_object_type(validate_inferred_attr_types, bool, 'Validate inferred attribute type')

        # Get the similarity functions to be used for matching
        sim_funcs = NewFeatures.get_supported_sim_funs()
        # Get the tokenizers to be used for matching
        tok_funcs = tok.get_tokenizers_for_matching(q=[2,3,4], 
                                                    # dlm_char=[[' ', '.', ',', '\\', '\t', '\r', '\n']]
                                                    dlm_char=[[' ', '\'', '\"', ',', '\\', '\t', '\r', '\n']])
        print(tok_funcs)

        # Get the attr. types for ltable and rtable
        attr_types_ltable = au.get_attr_types(ltable)
        attr_types_rtable = au.get_attr_types(rtable)
        aindex = list(ltable)
        bindex = list(rtable)

        # Fix the attr type.
        # This is essential for generating features
        # Feature generated according to appendix in "Sigmod2017-Falcon"
        if at_ltable is None and at_rtable is None:
            type_length = len(attr_types_ltable) - 1
            for i in range(1, type_length):
                # not consistent in l and r
                if attr_types_ltable[aindex[i]] < attr_types_rtable[bindex[i]]:
                    print("\033[31mFix:\033[0m", attr_types_rtable[bindex[i]], "to", attr_types_ltable[aindex[i]], "for", aindex[i])
                    attr_types_rtable[bindex[i]] = attr_types_ltable[aindex[i]]
                elif attr_types_ltable[aindex[i]] > attr_types_rtable[bindex[i]]:
                    print("\033[31mFix:\033[0m", attr_types_ltable[aindex[i]], "to", attr_types_rtable[bindex[i]], "for", aindex[i])
                    attr_types_ltable[aindex[i]] = attr_types_rtable[bindex[i]]
                # some specific attribute
                if (aindex[i] == 'brand' or aindex[i] == 'category') and dataname == "secret":
                    attr_types_ltable[aindex[i]] = 'str_eq_1w'
                    attr_types_rtable[bindex[i]] = 'str_eq_1w'
                elif aindex[i] == 'description' and dataname == "secret":
                    attr_types_ltable[aindex[i]] = 'str_gt_10w'
                    attr_types_rtable[bindex[i]] = 'str_gt_10w'
        else:
            attr_types_ltable = at_ltable if at_ltable is not None else attr_types_ltable
            attr_types_rtable = at_rtable if at_rtable is not None else attr_types_rtable

        # Get the attr. correspondences between ltable and rtable
        attr_corres = au.get_attr_corres(ltable, rtable)
        # print(attr_types_ltable)
        # print(attr_corres)

        # Get features based on attr types, attr correspondences, sim functions
        # and tok. functions
        feature_table = autog.get_features(ltable, rtable, attr_types_ltable,
                                           attr_types_rtable, attr_corres,
                                           tok_funcs, sim_funcs)

        # Export important variables to global name space
        em._block_t = tok_funcs
        em._block_s = sim_funcs
        em._atypes1 = attr_types_ltable
        em._atypes2 = attr_types_rtable
        em._block_c = attr_corres
        # Return the feature table
        
        return feature_table


'''
current deprecated because of effciency
can be used in testing
'''
class NewFeatureExtractor:
    '''
    re-write a feature extractor from the py_entitymatching module
    support interchangeable values for calculating values
    '''

    def __init__(self):
        pass
    

    @staticmethod
    def _get_num_procs(n_jobs, min_procs):
        # determine number of processes to launch parallely
        n_cpus = multiprocessing.cpu_count()
        n_procs = n_jobs
        if n_jobs < 0:
            n_procs = n_cpus + 1 + n_jobs
        # cannot launch less than min_procs to safeguard against small tables
        return min(n_procs, min_procs)


    @staticmethod
    def _apply_feat_fns(tuple1, tuple2, feat_dict, group, cluster):
        """
        Apply feature functions to two tuples, consider interchangeable values
        """
        # Get the feature names
        feat_names = list(feat_dict['feature_name'])
        feat_attrs = [feat_name.split("_")[0] for feat_name in feat_names]
        # Get the feature functions
        feat_funcs = list(feat_dict['function'])
        # Compute the feature value by applying the feature function to the input
        #  tuples.
        feat_vals = []
        for attr, func in zip(feat_attrs, feat_funcs):
            # such attr is not qualified to be evluated with interchangeable values
            # too short, e.g., numeric
            if attr not in group:
                feat_vals.append(func(tuple1, tuple2))
            else:
                attr_grp = group[attr]
                attr_clt = cluster[attr]

                # id1, id2 = tuple1["id"], tuple2["id"]
                oval1, oval2 = tuple1[attr], tuple2[attr] 
                
                has_key1 = oval1 in attr_clt
                has_key2 = oval2 in attr_clt
                if not has_key1 and not has_key2:
                    feat_vals.append(func(tuple1, tuple2))
                    continue
                elif has_key1 and not has_key2:
                    max_val = -1.0
                    key1 = attr_clt[oval1]
                    for val1 in attr_grp[key1]:
                        tuple1.loc[attr] = val1
                        fval = func(tuple1, tuple2)
                        max_val = fval if fval > max_val else max_val
                    feat_vals.append(max_val)
                    tuple1.loc[attr]= oval1
                    if max_val < 0:
                        raise ValueError("error occurs in calculating features for interchangeable values")
                    continue
                elif not has_key1 and has_key2:
                    max_val = -1.0
                    key2 = attr_clt[oval2]
                    for val2 in attr_grp[key2]:
                        tuple2.loc[attr] = val2
                        fval = func(tuple1, tuple2)
                        max_val = fval if fval > max_val else max_val
                    feat_vals.append(max_val)
                    tuple2.loc[attr] = oval2
                    if max_val < 0:
                        raise ValueError("error occurs in calculating features for interchangeable values")
                    continue

                key1, key2 = attr_clt[oval1], attr_clt[oval2]
                if key1 == key2:
                    feat_vals.append(func(tuple1, tuple1))
                else:
                    max_val = -1.0
                    for val1 in attr_grp[key1]:
                        for val2 in attr_grp[key2]:
                            tuple1.loc[attr], tuple2.loc[attr] = val1, val2
                            fval = func(tuple1, tuple2)
                            max_val = fval if fval > max_val else max_val
                    if max_val < 0:
                        raise ValueError("error occurs in calculating features for interchangeable values")
                    feat_vals.append(max_val)
                    tuple1.loc[attr], tuple2.loc[attr] = oval1, oval2
                    # oval = func(tuple1, tuple2)
                    # if max_val > oval:
                    #     print(f"find: {max_val}, {oval}")
                
        # feat_vals = [f(tuple1, tuple2) for f in feat_funcs]
        # print(feat_vals)
        # print([f(tuple1, tuple2) for f in feat_funcs])
        # Return a dictionary where the keys are the feature names and the values
        #  are the feature values.
        return dict(zip(feat_names, feat_vals))


    @staticmethod
    def _get_feature_vals_by_cand_split(pickled_obj, pickled_grp, pickled_clt, fk_ltable_idx, fk_rtable_idx, l_df, r_df, candsplit):
        feature_table = cloudpickle.loads(pickled_obj)
        group = cloudpickle.loads(pickled_grp)
        cluster = cloudpickle.loads(pickled_clt)

        l_dict = {}
        r_dict = {}

        feat_vals = []
        for row in candsplit.itertuples(index=False):
            fk_ltable_val = row[fk_ltable_idx]
            fk_rtable_val = row[fk_rtable_idx]

            # return the reference rather than copy
            if fk_ltable_val not in l_dict:
                l_dict[fk_ltable_val] = l_df.loc[fk_ltable_val].copy()
            l_tuple = l_dict[fk_ltable_val]

            if fk_rtable_val not in r_dict:
                r_dict[fk_rtable_val] = r_df.loc[fk_rtable_val].copy()
            r_tuple = r_dict[fk_rtable_val]

            f = NewFeatureExtractor._apply_feat_fns(l_tuple, r_tuple, feature_table, group, cluster)
            feat_vals.append(f)

        return feat_vals


    @staticmethod
    def _extract_from(candset, feature_table, group, cluster, n_jobs, verbose=False):
        # Get metadata for candidate set
        key, fk_ltable, fk_rtable, ltable, rtable, l_key, r_key = \
            cm.get_metadata_for_candset(candset, logger, verbose)
        
        # Set index for convenience
        l_df = ltable.set_index(l_key, drop=False)
        r_df = rtable.set_index(r_key, drop=False)
        
        # Apply feature functions
        ch.log_info(logger, 'Applying feature functions', verbose)
        col_names = list(candset.columns)
        fk_ltable_idx = col_names.index(fk_ltable)
        fk_rtable_idx = col_names.index(fk_rtable)

        n_procs = NewFeatureExtractor._get_num_procs(n_jobs, len(candset))

        c_splits = np.array_split(candset, n_procs)

        pickled_obj = cloudpickle.dumps(feature_table)
        pickled_grp = cloudpickle.dumps(group)
        pickled_clt = cloudpickle.dumps(cluster)

        feat_vals_by_splits = Parallel(n_jobs=n_procs)(
            delayed(NewFeatureExtractor._get_feature_vals_by_cand_split)(
                pickled_obj,
                pickled_grp,
                pickled_clt,
                fk_ltable_idx,
                fk_rtable_idx,
                l_df,
                r_df,
                c_split
            )
            for i, c_split in enumerate(c_splits)
        )

        feat_vals = sum(feat_vals_by_splits, [])
        return feat_vals


    @staticmethod
    def extract_feature_vecs(candset, attrs_before=None, feature_table=None,
                             attrs_after=None, group=None, cluster=None,
                             verbose=False, n_jobs=1):
        # (Matt) Stage 1: Input validation
        # Validate input parameters

        # # We expect the input candset to be of type pandas DataFrame.
        validate_object_type(candset, pd.DataFrame, error_prefix='Input cand.set')

        # (Matt) The two blocks below are making sure that attributes that are to be appended
        # to this function's output do in fact exist in the input DataFrame
        
        # # If the attrs_before is given, Check if the attrs_before are present in
        # the input candset
        if attrs_before != None:
            if not ch.check_attrs_present(candset, attrs_before):
                logger.error(
                    'The attributes mentioned in attrs_before is not present '
                    'in the input table')
                raise AssertionError(
                    'The attributes mentioned in attrs_before is not present '
                    'in the input table')

        # # If the attrs_after is given, Check if the attrs_after are present in
        # the input candset
        if attrs_after != None:
            if not ch.check_attrs_present(candset, attrs_after):
                logger.error(
                    'The attributes mentioned in attrs_after is not present '
                    'in the input table')
                raise AssertionError(
                    'The attributes mentioned in attrs_after is not present '
                    'in the input table')

        # (Matt) Why not make sure that this is a DataFrame instead of just nonempty?
        # We expect the feature table to be a valid object
        if feature_table is None:
            logger.error('Feature table cannot be null')
            raise AssertionError('The feature table cannot be null')

        # Do metadata checking
        # # Mention what metadata is required to the user
        ch.log_info(logger, 'Required metadata: cand.set key, fk ltable, '
                            'fk rtable, '
                            'ltable, rtable, ltable key, rtable key', verbose)

        # (Matt) ch ~ catalog helper
        # # Get metadata
        ch.log_info(logger, 'Getting metadata from catalog', verbose)

        # (Matt) cm ~ catalog manager
        key, fk_ltable, fk_rtable, ltable, rtable, l_key, r_key = \
            cm.get_metadata_for_candset(
                candset, logger, verbose)

        # # Validate metadata
        ch.log_info(logger, 'Validating metadata', verbose)
        cm._validate_metadata_for_candset(candset, key, fk_ltable, fk_rtable,
                                        ltable, rtable, l_key, r_key,
                                        logger, verbose)

        # Extract features
        # # Apply feature functions
        feat_vals = NewFeatureExtractor._extract_from(candset, feature_table, group, cluster, n_jobs, verbose)
        
        # (Matt) ParallelFeatureExtractor implementation ends here; the rest is formatting

        # Construct output table
        feature_vectors = pd.DataFrame(feat_vals, index=candset.index.values)
        # # Rearrange the feature names in the input feature table order
        feature_names = list(feature_table['feature_name'])
        feature_vectors = feature_vectors[feature_names]

        ch.log_info(logger, 'Constructing output table', verbose)
        # print(feature_vectors)
        # # Insert attrs_before
        if attrs_before:
            if not isinstance(attrs_before, list):
                attrs_before = [attrs_before]
            attrs_before = gh.list_diff(attrs_before, [key, fk_ltable, fk_rtable])
            attrs_before.reverse()
            for a in attrs_before:
                feature_vectors.insert(0, a, candset[a])

        # # Insert keys
        feature_vectors.insert(0, fk_rtable, candset[fk_rtable])
        feature_vectors.insert(0, fk_ltable, candset[fk_ltable])
        feature_vectors.insert(0, key, candset[key])

        # # insert attrs after
        if attrs_after:
            if not isinstance(attrs_after, list):
                attrs_after = [attrs_after]
            attrs_after = gh.list_diff(attrs_after, [key, fk_ltable, fk_rtable])
            attrs_after.reverse()
            col_pos = len(feature_vectors.columns)
            for a in attrs_after:
                feature_vectors.insert(col_pos, a, candset[a])
                col_pos += 1

        # Reset the index
        # feature_vectors.reset_index(inplace=True, drop=True)

        # # Update the catalog
        cm.init_properties(feature_vectors)
        cm.copy_properties(candset, feature_vectors)

        # Finally, return the feature vectors
        return feature_vectors