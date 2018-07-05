# !/bin/bash

echo ------------------------------------------------------------------------------
echo STEP 1: Install the environment for running dehazenet.
echo ------------------------------------------------------------------------------
pip3 install numpy
pip3 install PILLOW
pip3 install threadpool
echo ------------------------------------------------------------------------------
echo STEP 2: Creating directory for dehazenet
echo ------------------------------------------------------------------------------
CURRENT_FILE_DIR=$(cd `dirname $0`; pwd)
# Please enter your desired root path as follow.
PROGRAM_ROOT_PATH='/home/xiaob6/test'
if [ ! -d '$PROGRAM_ROOT_PATH' ];then
    mkdir $PROGRAM_ROOT_PATH
else
	rm -rf $PROGRAM_ROOT_PATH
fi
cd $PROGRAM_ROOT_PATH
echo [$PROGRAM_ROOT_PATH]: Root directory for current project.
mkdir ClearImages
echo [$PROGRAM_ROOT_PATH/ClearImages]: Save clear images.
cd ./ClearImages
mkdir TestImages
echo [$PROGRAM_ROOT_PATH/TestImages]: Save clear test images.
mkdir TrainImages
echo [$PROGRAM_ROOT_PATH/TrainImages]: Save clear training images.
mkdir TransImages
echo [$PROGRAM_ROOT_PATH/TransImages]: Save transmission maps.
cd ..
mkdir ClearResultImages
echo [$PROGRAM_ROOT_PATH/ClearResultImages]: Save result images.
mkdir HazeImages
echo [$PROGRAM_ROOT_PATH/HazeImages]: Save haze images.
cd ./ClearImages
mkdir TestImages
echo [$PROGRAM_ROOT_PATH/TestImages]: Save haze test images.
mkdir TrainImages
echo [$PROGRAM_ROOT_PATH/TestImages]: Save haze training images.
cd ..
mkdir StatisticalFigure
echo [$PROGRAM_ROOT_PATH/StatisticalFigure]: Save haze training images.
mkdir TFRecord
echo [$PROGRAM_ROOT_PATH/TFRecord]: Save tfrecords for training.
mkdir StatisticalResults
echo [$PROGRAM_ROOT_PATH/StatisticalResults]: Save Statistical Result for evaluating results.
mkdir DeHazeNetModel
echo [$PROGRAM_ROOT_PATH/DeHazeNetModel]: Save training model.
echo ------------------------------------------------------------------------------
echo STEP 3: Copying files to your desired root path.
echo ------------------------------------------------------------------------------
cd $CURRENT_FILE_DIR
cp -rf * $PROGRAM_ROOT_PATH
cd $PROGRAM_ROOT_PATH
echo Please Refers to README.md for further information.
echo ------------------------------------------------------------------------------
echo Congratulations! Finish setup!
echo ------------------------------------------------------------------------------
