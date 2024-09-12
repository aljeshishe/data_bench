import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
import socket
import time
import boto3
import logging

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
instance_params = """{
  "MaxCount": 1,
  "MinCount": 1,
  "ImageId": "ami-07b52a7a38d09d6f7",
  "InstanceType": "g4dn.12xlarge",
  "KeyName": "grachev_ab_aws_key",
  "EbsOptimized": true,
  "NetworkInterfaces": [
    {
      "SubnetId": "subnet-0604a4cc2fedcce55",
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
  "InstanceMarketOptions": {
    "MarketType": "spot"
  },
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

def main():
    response = client.run_instances(**json.loads(instance_params))
    time.sleep(5)
    instance_id = response['Instances'][0]['InstanceId']
    print(f'Launched instance with ID: {instance_id}')
    instance_description = client.describe_instances(InstanceIds=[instance_id])
    instance_details = instance_description['Reservations'][0]['Instances'][0]
    ip = instance_details["PrivateIpAddress"]
    print(ip)


        
if __name__ == "__main__":
    main()
    