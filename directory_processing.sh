# Starting in the directory containing thruxton_group_all

# Create the directories for rgb_imgs and segm_imgs_bin outside of thruxton_group_all
mkdir -p train
mkdir -p val
mkdir -p test
mkdir -p train_mask
mkdir -p val_mask
mkdir -p test_mask

echo $(pwd)

# Get the shuffled list of image filenames without extension
files=$(ls /content/thurxton_group_all/thurxton_group/rgb_imgs | shuf)
total_files=$(echo "$files" | wc -l)
train_count=$(( total_files * 80 / 100 ))
val_count=$(( total_files * 10 / 100 ))
test_count=$(( total_files - train_count - val_count ))

# Move the files for both rgb_imgs and segm_imgs_bin based on the filenames
echo "$files" | head -n $train_count | while read file; do
    mv "thurxton_group_all/thurxton_group/rgb_imgs/$file" ./train/
    mv "thurxton_group_all/thurxton_group/segm_imgs_bin/$file" ./train_mask/
done

echo "$files" | head -n $(($train_count + $val_count)) | tail -n $val_count | while read file; do
    mv "thurxton_group_all/thurxton_group/rgb_imgs/$file" ./val/
    mv "thurxton_group_all/thurxton_group/segm_imgs_bin/$file" ./val_mask/
done

echo "$files" | tail -n $test_count | while read file; do
    mv "thurxton_group_all/thurxton_group/rgb_imgs/$file" ./test/
    mv "thurxton_group_all/thurxton_group/segm_imgs_bin/$file" ./test_mask/
done
