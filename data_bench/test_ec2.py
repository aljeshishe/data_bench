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
instance_params = """{
  "MaxCount": 1,
  "MinCount": 1,
  "ImageId": "ami-07b52a7a38d09d6f7",
  "InstanceType": "c5.18xlarge",
  "KeyName": "grachev_ab_aws_key",
  "EbsOptimized": true,
  "BlockDeviceMappings": [
    {
      "DeviceName": "/dev/xvda",
      "Ebs": {
        "Encrypted": false,
        "DeleteOnTermination": true,
        "Iops": 3000,
        "SnapshotId": "snap-017e22242ecff9455",
        "VolumeSize": 500,
        "VolumeType": "gp3",
        "Throughput": 125
      }
    }
  ],
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

def test(i):
    start = time.time()
    # Launch the EC2 instance
    response = client.run_instances(**json.loads(instance_params))
    instance_id = response['Instances'][0]['InstanceId']
    print(f'Launched instance with ID: {instance_id}')

    time.sleep(5)

    try:
        instance_description = client.describe_instances(InstanceIds=[instance_id])
        instance_details = instance_description['Reservations'][0]['Instances'][0]
        ip = instance_details["PrivateIpAddress"]
        print(f"Connecting to {ip}")
        while True:
            with suppress(Exception):
                with socket.create_connection((ip, 22)) as conn:
                    elapsed = time.time() - start
                    return elapsed
            time.sleep(1)
    finally:
        print("Terminating instance")
        client.terminate_instances(InstanceIds=[instance_id])
        
def run_tests(count=6):
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(test, range(count)))
    results = [round(r, 2) for r in sorted(results)]
    print(f"Results: {results}")
    print(f"Min: {min(results)}")
    print(f"Avg: {sum(results) / len(results)}")    
    print(f"Max: {max(results)}")
        
if __name__ == "__main__":
    run_tests()
    