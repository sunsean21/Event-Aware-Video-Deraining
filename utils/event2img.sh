#!/bin/bash

catagories=(Dataset_Testing_RealRain
Dataset_Training_Synthetic
Dataset_Testing_Synthetic)

working_path=~/Desktop/phd/proj1/SPAC-SupplementaryMaterials
cur_path=`pwd`
for catagory in "${catagories[@]}"
do
    mkdir ../dataset/event/$catagory
    cd $working_path
    for folder in `find Extracted/$catagory/ -iname "*_Rain*"`
    do 
        echo "Processing $folder"
        name_purflex=`echo $folder | sed 's/\//_/g'`        
        # get frame_size
        # framesize=`identify -format '%w %h' $folder/00001.jpg`
        # OLD_IFS="$IFS"
        # IFS=" "
        # framesize=($framesize)
        
        # split path
        OLD_IFS="$IFS"
        IFS="/"
        folder_path=($folder)
        IFS="$OLD_IFS"

        cd $cur_path
        mkdir ../dataset/event/$catagory/${folder_path[-1]}

        python ./event2img.py \
        --path $working_path/outbags/out_$name_purflex.bag \
        --out_path ../dataset/event/${folder_path[-2]}/${folder_path[-1]}
        # --width ${framesize[0]} \
        # --height ${framesize[1]} \
        
    done
done
