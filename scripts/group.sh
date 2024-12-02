#! /bin/bash

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -ga|--group_attr)
      GROUP_ATTR="$2"
      shift # past argument
      shift # past value
      ;;
    -gs|--group_strategy)
      GROUP_STRATEGY="$2"
      shift # past argument
      shift # past value
      ;;
    -idir|--icv_dir)
      ICV_DIR="$2"
      shift # past argument
      shift # past value
      ;;
    -tau|--group_tau)
      GROUP_TAU="$2"
      shift # past argument
      shift # past value
      ;;
    -tc|--transitive_closure)
      TC="$2"
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

echo "Group Attribute                 : ${GROUP_ATTR}"
echo "Group Strategy                  : ${GROUP_STRATEGY}"
echo "Default ICV Directory           : ${ICV_DIR}"
echo "Group Tau                       : ${GROUP_TAU}"
echo "Is Transitive Closure           : ${TC}"

##############################
# change this line if needed #
##############################
bin/sample $BLOCKING_ATTR $NUM_DATA $SAMPLE_STRATEGY ${POSITIONAL_ARGS[@]} "$PATH_TABLE_A" "$PATH_TABLE_B" "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo "done sample."
else
    echo "fail to sample."
fi