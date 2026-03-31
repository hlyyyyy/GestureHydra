CONFIG=$1
WORK_DIR=$2

PYTHONPATH=".":$PYTHONPATH python tools/train.py $CONFIG --work-dir $WORK_DIR
# PORT=29751 bash ./tools/dist_train.sh $CONFIG $WORK_DIR 4
