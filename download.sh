#! /bin/sh
set -e
GDRIVE_ID=0BxYys69jI14kYVM3aVhKS1VhRUk
echo Downloading file...
curl -Ls "https://docs.google.com/uc?export=download&confirm=t&id=$GDRIVE_ID" -o UTKFace.tar.gz
echo Extracting file...
tar xzf UTKFace.tar.gz
echo Cleaning up files...
rm UTKFace.tar.gz
echo All Done! You should now run preprocessing.py to get the data ready for the model
