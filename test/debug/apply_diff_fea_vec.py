import py_entitymatching as em


def apply_model_debug(rf, tableA, tableB):
    # Evaluate the predictions
    H1 = em.read_csv_metadata("test/debug/ori_fea_vec.csv", 
                                key="_id", 
                                ltable=tableA, rtable=tableB,
                                fk_ltable="ltable_id", fk_rtable="rtable_id")
    H2 = em.read_csv_metadata("test/debug/pro_fea_vec.csv", 
                                key="id", 
                                ltable=tableA, rtable=tableB,
                                fk_ltable="ltable_id", fk_rtable="rtable_id")
    
    rf.label_cand(H1)
    rf.label_cand(H2)

    predictions1 = rf.rf.predict(table=H1, exclude_attrs=['_id', 'ltable_id', 'rtable_id', 'label'], 
                                 append=True, target_attr='predicted', inplace=True, 
                                 return_probs=True, probs_attr='proba')
    predictions2 = rf.rf.predict(table=H2, exclude_attrs=['id', 'ltable_id', 'rtable_id', 'label'], 
                                 append=True, target_attr='predicted', inplace=True, 
                                 return_probs=True, probs_attr='proba')

    # eval_result = em.eval_matches(predictions, 'label', 'predicted')
    # em.print_eval_summary(eval_result)

    print("report the diff results .....")
    eval_result = em.eval_matches(predictions1, 'label', 'predicted')
    em.print_eval_summary(eval_result)
    eval_result = em.eval_matches(predictions2, 'label', 'predicted')
    em.print_eval_summary(eval_result)
    print("diff results done .....")
