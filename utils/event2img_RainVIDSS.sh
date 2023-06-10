#!/bin/bash

catagories=(
    rainy
    )

working_path=~/Desktop/phd/proj1/RainVIDSS/dataset_RainVIDSS
cur_path=`pwd`
for catagory in "${catagories[@]}"
do
    mkdir ../dataset/event/RainVIDSS/$catagory
    cd $working_path
    for folder in `find val/$catagory/ -iname "*seq_*"`
    do 
        cd $working_path
        echo "Processing $folder"
        name_purflex=`echo $folder | sed 's/\//_/g'`        
        # get frame_size
        framesize=`identify -format '%w %h' $folder/0000.jpg`
        OLD_IFS="$IFS"
        IFS=" "
        framesize=(${framesize})
        IFS="$OLD_IFS"
        echo ${framesize[0]}
        echo ${framesize[1]}
        # split path
        OLD_IFS="$IFS"
        IFS="/"
        folder_path=($folder)
        IFS="$OLD_IFS"

        cd $cur_path
        mkdir ../dataset/event/RainVIDSS/$catagory/${folder_path[-1]}

        ~/anaconda3/bin/python3 ./event2img.py \
        --path $working_path/outbags/out_$name_purflex.bag \
        --out_path ../dataset/event/RainVIDSS/${folder_path[-2]}/${folder_path[-1]} \
        --width ${framesize[0]} \
        --height ${framesize[1]} \
        
    done
done
