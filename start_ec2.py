import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
import socket
import time
import boto3
import logging

from data_bench import ec2_utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
session = boto3.Session()

# Create an EC2 client
client = session.client('ec2')

# Define the EC2 instance parameters
#
# ami-00c79d83cf718a893 Amazon linux 2023 13 secs
# ami-0cab37bd176bb80d3 ubuntu 24.04 15-20 secs
# ami-0162fe8bfebb6ea16 ubuntu 22.04 16-17 secs
# ami-0b91d972a18405725 ubuntu 22.04 dlami
# ami-07b52a7a38d09d6f7 ubuntu 22.04 custom alber blanc

# ami_id: ami-01ce10261fc6c0923 # Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)
# ami-05365f11230f74538 # dlami+stuff
# ami-0d03d127597f4c4d7 # ubuntu_22+stuff

# aval zones
# |  ll-subnet-public2-ap-northeast-1a |  subnet-04848cdc1b122a8a6  |  ap-northeast-1a |  apne1-az3 |
# |  ll-subnet-public3-ap-northeast-1c |  subnet-0dc12960b2d51bad8  |  ap-northeast-1c |  apne1-az1 |
# |  ll-subnet-public1-ap-northeast-1d |  subnet-0604a4cc2fedcce55  |  ap-northeast-1d |  apne1-az2 |


AVAILABLE_ZONE = "ap-northeast-1b"
SIZE_GB = 200
instance_params = """{
  "MaxCount": 1,
  "MinCount": 1,
  "ImageId": "ami-08f191dd81ec3a3de",
  "InstanceType": "g6.8xlarge",
  "KeyName": "grachev_ab_aws_key",
  "EbsOptimized": true,
  "NetworkInterfaces": [
    {
      "SubnetId": "subnet-0e8b8c23ddb1ada49",
      "AssociatePublicIpAddress": true,
      "DeviceIndex": 0,
      "Groups": [
        "sg-0004eeb822745ac47"
      ]
    }
  ],
  "TagSpecifications": [
    {
      "ResourceType": "instance",
      "Tags": [
        {
          "Key": "Name",
          "Value": "tmp-grachev"
        }
      ]
    }
  ],
  "BlockDeviceMappings": [
    {
      "DeviceName": "/dev/sda1",
      "Ebs": {
        "Encrypted": false,
        "DeleteOnTermination": true,
        "Iops": 3000,
        "SnapshotId": "snap-0690f32038e7a1dd2",
        "VolumeSize": 300,
        "VolumeType": "gp3"
      }
    }
  ],  
  "MetadataOptions": {
    "HttpEndpoint": "enabled",
    "HttpPutResponseHopLimit": 2,
    "HttpTokens": "required"
  },
  "PrivateDnsNameOptions": {
    "HostnameType": "ip-name",
    "EnableResourceNameDnsARecord": false,
    "EnableResourceNameDnsAAAARecord": false
  }
}"""
  # "InstanceMarketOptions": {
  #   "MarketType": "spot",
  #   "SpotOptions": {
  #     "MaxPrice": "10"
  #   }  
  # },

def main():
    params = json.loads(instance_params)
    params["NetworkInterfaces"][0]["SubnetId"] = "subnet-0e8b8c23ddb1ada49" # ec2_utils.get_subnet_id(availability_zone=AVAILABLE_ZONE)
    params["BlockDeviceMappings"] = ec2_utils.get_block_device_mappings(size_gb=SIZE_GB, params=params)
    response = client.run_instances(**params)
    time.sleep(5)
    instance_id = response['Instances'][0]['InstanceId']
    print(f'Launched instance with ID: {instance_id}')
    instance_description = client.describe_instances(InstanceIds=[instance_id])
    instance_details = instance_description['Reservations'][0]['Instances'][0]
    ip = instance_details["PrivateIpAddress"]
    print(ip)


        
if __name__ == "__main__":
    main()
    