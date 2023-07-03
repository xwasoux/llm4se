cd ../data

wget https://dax-cdn.cdn.appdomain.cloud/dax-project-codenet/1.0.0/Project_CodeNet_Python800.tar.gz
wget https://dax-cdn.cdn.appdomain.cloud/dax-project-codenet/1.0.0/Project_CodeNet_Python800_spts.tar.gz
wget https://dax-cdn.cdn.appdomain.cloud/dax-project-codenet/1.0.0/Project_CodeNet_Python800_cass.tar.gz
wget https://dax-cdn.cdn.appdomain.cloud/dax-project-codenet/1.0.0/Project_CodeNet_metadata.tar.gz

tar -zxvf Project_CodeNet_Python800.tar.gz
tar -zxvf Project_CodeNet_Python800_spts.tar.gz
tar -zxvf Project_CodeNet_Python800_cass.tar.gz
tar -zxvf Project_CodeNet_metadata.tar.gz

rm Project_CodeNet_Python800.tar.gz
rm Project_CodeNet_Python800_spts.tar.gz
rm Project_CodeNet_Python800_cass.tar.gz
rm Project_CodeNet_metadata.tar.gz

cd ..
