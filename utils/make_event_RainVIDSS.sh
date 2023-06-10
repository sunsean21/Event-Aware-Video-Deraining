#!/bin/bash
catagories=(
    rainy)

working_path=~/Desktop/phd/proj1/RainVIDSS/dataset_RainVIDSS

for catagory in "${catagories[@]}"
do
    for folder in `find val/$catagory/ -iname "seq_*"`
    do
        # ssim
        # roscd esim_ros
        if [[ "${folder}" =~ "seq_long_" ]]; then
            frequency=1
        else
            frequency=30
        fi
        echo $frequency
        python ~/sim_ws/src/rpg_esim/event_camera_simulator/esim_ros/scripts/generate_stamps_file_jpg.py -i $working_path/$folder -r $frequency

        name_purflex=`echo $folder | sed 's/\//_/g'`

        rosrun esim_ros esim_node \
         --data_source=2 \
         --path_to_output_bag=$working_path/outbags/out_$catagory_$name_purflex.bag \
         --path_to_data_folder=$working_path/$folder \
         --ros_publisher_frame_rate=60 \
         --exposure_time_ms=10.0 \
         --use_log_image=1 \
         --log_eps=0.1 \
         --contrast_threshold_pos=0.15 \
         --contrast_threshold_neg=0.15        
    done
done


echo "for the later visualization, please go to https://github.com/uzh-rpg/rpg_esim/wiki/Simulating-events-from-a-video"