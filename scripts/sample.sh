#! /bin/bash

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -st|--sample_strategy)
      SAMPLE_STRATEGY="$2"
      shift # past argument
      shift # past value
      ;;
    -ba|--blocking_attr)
      BLOCKING_ATTR="$2"
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
    -odir|--output_dir)
      OUTPUT_DIR="$2"
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

echo "Num Data                        : ${NUM_DATA}"
echo "Sample Strategy                 : ${SAMPLE_STRATEGY}"
echo "Blocking(working) Attribute     : ${BLOCKING_ATTR}"
echo "Table A Path                    : ${PATH_TABLE_A}"
echo "Table B Path                    : ${PATH_TABLE_B}"
echo "Output Dir Path                 : ${OUTPUT_DIR}"
echo "Customized Arguments            : ${POSITIONAL_ARGS[@]}"

##############################
# change this line if needed #
##############################
bin/sample $BLOCKING_ATTR $NUM_DATA $SAMPLE_STRATEGY ${POSITIONAL_ARGS[@]} "$PATH_TABLE_A" "$PATH_TABLE_B" "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo "done sample."
else
    echo "fail to sample."
fi