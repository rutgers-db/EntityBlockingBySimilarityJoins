We directly modify the following parts of the **py_entitymatching**, such modified package can be found on GitHub: [py_entitymatching](https://github.com/EricLYunqi/py_entitymatching/tree/master).

1. On py_entitymatching, in <span style="color:orange">feature/tokenizer.py</span>. We change the function "**_make_tok_delim**", add the case that "d" is a list.
2. On py_entitymatching, in <span style="color:orange">blocker/rule_based_blocker.py</span>. We change the function "**block_tables**", add the case to remove "PARSE_EXP" in "l_proj_attrs" as well as "r_proj_attrs".
3. On py_entitymatching, in <span style="color:orange">feature/autofeaturegem.py</span>. We change the function "**_get_feat_lkp_tbl**", modify the default generated features for all types of attributes.
4. On py_entitymatching, in <span style="color:orange">feature/autofeaturegen.py</span>. We change the function "**conv_fn_str_to_obj**", add the case to check if "f" in "fn_tup" is None.

Also for **py_stringmatching**:
1. On py_stringmatching, in <span style="color:orange">tokenizer/qgram_tokenizer</span>. We change the function "**tokenize**" under the class "**QgramTokenizer**" to eliminate the q-gram tokens which are not alphanumeric

At last, it is worth to mention that we installed Megallen packages on serval different **Linux** meachines and found it may fail due to following reasons:
1. **py_stringmatching** needs numpy < 0.2.0 due to a new recent release.
2. **py_stringsimjoin** can only be installed by building source code since the source release on PyPI is missing cythonize.py (*not sure if it is fixed now, please try pip install first*).
