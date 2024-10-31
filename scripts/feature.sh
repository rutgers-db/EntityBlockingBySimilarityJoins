#! /bin/bash

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -u|--usage)
      USAGE="$2"
      shift # past argument
      shift # past value
      ;;
    -inter|--is_interchangeable)
      IS_INTERCHANGEABLE="$2"
      shift # past argument
      shift # past value
      ;;
    -con|--flag_consistent)
      FLAG_CONSISTENT="$2"
      shift # past argument
      shift # past value
      ;;
    -ntab|--total_table)
      TOTAL_TABLE="$2"
      shift # past argument
      shift # past value
      ;;
    -nattr|--total_attr)
      TOTAL_ATTR="$2"
      shift # past argument
      shift # past value
      ;;
    -feavecd|--feature_vec_dir)
      FEATURE_VEC_DIR="$2"
      shift # past argument
      shift # past value
      ;;
    -icvd|--icv_dir)
      ICV_DIR="$2"
      shift # past argument
      shift # past value
      ;;
    -feanamed|--feature_names_dir)
      FEATURE_NAMES_DIR="$2"
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

echo "Usage                                    : ${USAGE}"
echo "Is Interchangeable                       : ${IS_INTERCHANGEABLE}"
echo "Flag Consistent                          : ${FLAG_CONSISTENT}"
echo "Total Table                              : ${TOTAL_TABLE}"
echo "Total Attributes                         : ${TOTAL_ATTR}"
echo "Attributes                               : ${POSITIONAL_ARGS[@]}"
echo "Default Feature Vectors Directory:       : ${FEATURE_VEC_DIR}"
echo "Default Interchangeable Value Directory  : ${ICV_DIR}"
echo "Default Feature Names Directory:         : ${FEATURE_NAMES_DIR}"
echo "The attributes not included above will be calculated by default"

##############################
# change this line if needed #
##############################
bin/feature $USAGE $IS_INTERCHANGEABLE $FLAG_CONSISTENT $TOTAL_TABLE $TOTAL_ATTR ${POSITIONAL_ARGS[@]} \
            "${FEATURE_VEC_DIR}" "${ICV_DIR}" "${FEATURE_NAMES_DIR}"

if [ $? -eq 0 ]; then
    echo "done feature."
else
    echo "fail to feature."
fi