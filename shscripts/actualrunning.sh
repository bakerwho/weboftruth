git pull

pip install -e .

nohup bash shscripts/awsmodels.sh >./logs/aws.out 2>./logs/aws.err &

tail -20 ./logs/aws.err
