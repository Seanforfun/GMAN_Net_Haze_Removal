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
echo $PROGRAM_ROOT_PATH is root directory for current project.
mkdir ClearImages
echo $PROGRAM_ROOT_PATH/ClearImages is used to save clear images.
cd ./ClearImages
mkdir TestImages
echo $PROGRAM_ROOT_PATH/TestImages is used to save clear test images.
mkdir TrainImages
echo $PROGRAM_ROOT_PATH/TrainImages is used to save clear training images.
mkdir TransImages
echo $PROGRAM_ROOT_PATH/TransImages is used to save transmission maps.
cd ..
mkdir ClearResultImages
echo $PROGRAM_ROOT_PATH/ClearResultImages is used to save result images.
mkdir HazeImages
echo $PROGRAM_ROOT_PATH/HazeImages is used to save haze images.
cd ./ClearImages
mkdir TestImages
echo $PROGRAM_ROOT_PATH/TestImages is used to save haze test images.
mkdir TrainImages
echo $PROGRAM_ROOT_PATH/TestImages is used to save haze training images.
cd ..
mkdir StatisticalFigure
echo $PROGRAM_ROOT_PATH/StatisticalFigure is used to save haze training images.
mkdir TFRecord
echo $PROGRAM_ROOT_PATH/TFRecord is used to save tfrecords for training.
mkdir StatisticalResults
echo $PROGRAM_ROOT_PATH/StatisticalResults is used to save Statistical Result for evaluating results.
mkdir DeHazeNetModel
echo $PROGRAM_ROOT_PATH/DeHazeNetModel is used to save training model.
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
