#! /bin/bash

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -jt|--join_type)
      JOIN_TYPE="$2"
      shift # past argument
      shift # past value
      ;;
    -js|--join_setting)
      JOIN_SETTING="$2"
      shift # past argument
      shift # past value
      ;;
    -ba|--blocking_attr)
      BLOCKING_ATTR="$2"
      shift # past argument
      shift # past value
      ;;
    -bat|--blocking_attr_type)
      BLOCKING_ATTR_TYPE="$2"
      shift # past argument
      shift # past value
      ;;
    -btopk|--blocking_top_k)
      BLOCKING_TOP_K="$2"
      shift # past argument
      shift # past value
      ;;
    -nd|--num_data)
      NUM_DATA="$2"
      shift # past argument
      shift # past value
      ;;
    -pathA|--path_table_A)
      PATH_TABLE_A="$2"
      shift # past argument
      shift # past value
      ;;
    -pathB|--path_table_B)
      PATH_TABLE_B="$2"
      shift # past argument
      shift # past value
      ;;
    -pathG|--path_gold)
      PATH_GOLD="$2"
      shift # past argument
      shift # past value
      ;;
    -pathR|--path_rule)
      PATH_RULE="$2"
      shift # past argument
      shift # past value
      ;;
    -odir|--output_dir)
      OUTPUT_DIR="$2"
      shift # past argument
      shift # past value
      ;;
    -sdir|--sample_res_dir)
      SAMPLE_RES_DIR="$2"
      shift # past argument
      shift # past value
      ;;
    -ts|--table_size)
      TABLE_SIZE="$2"
      shift # past argument
      shift # past value
      ;;
    -isjtopk|--is_join_topk)
      IS_JOIN_TOPK="$2"
      shift # past argument
      shift # past value
      ;;
    -isidfw|--is_idf_weighted)
      IS_IDF_WEIGHTED="$2"
      shift # past argument
      shift # past value
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters

echo "Join Type (0: self, 1: RS)                  : ${JOIN_TYPE}"
echo "Join Setting (0: serial, 1: parallel)       : ${JOIN_SETTING}"
echo "Num Data                                    : ${NUM_DATA}"
echo "Blocking(working) Attribute                 : ${BLOCKING_ATTR}"
echo "Blocking(working) Attribute Type            : ${BLOCKING_ATTR_TYPE}"
echo "Blocking Top K                              : ${BLOCKING_TOP_K}"
echo "Table A Path                                : ${PATH_TABLE_A}"
echo "Table B Path                                : ${PATH_TABLE_B}"
echo "Table Gold                                  : ${PATH_GOLD}"
echo "Table Rule                                  : ${PATH_RULE}"
echo "Table Size                                  : ${TABLE_SIZE}"
echo "Is Join TopK                                : ${IS_JOIN_TOPK}"
echo "Is IDF Weighted                             : ${IS_IDF_WEIGHTED}"
echo "Output Dir Path                             : ${OUTPUT_DIR}"
echo "Sample Result Dir                           : ${SAMPLE_RES_DIR}"
echo "Customized Arguments                        : ${POSITIONAL_ARGS[@]}"

##############################
# change this line if needed #
##############################
bin/block $JOIN_TYPE $JOIN_SETTING $BLOCKING_TOP_K $BLOCKING_ATTR $BLOCKING_ATTR_TYPE \
          "$PATH_TABLE_A" "$PATH_TABLE_B" "$PATH_GOLD" "$PATH_RULE" "$TABLE_SIZE" \
          $IS_JOIN_TOPK $IS_IDF_WEIGHTED "$OUTPUT_DIR" "$SAMPLE_RES_DIR"

if [ $? -eq 0 ]; then
    echo "done block."
else
    echo "fail to block."
fi