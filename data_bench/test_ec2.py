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
instance_params = {
    'MaxCount': 1,
    'MinCount': 1,
    'ImageId': 'ami-0162fe8bfebb6ea16',
    'InstanceType': 'r5.4xlarge',
    'KeyName': 'grachev_ab_aws_key',
    'EbsOptimized': False,
    'BlockDeviceMappings': [
        {
            'DeviceName': '/dev/xvda',
            'Ebs': {
                'Encrypted': False,
                'DeleteOnTermination': True,
                'SnapshotId': 'snap-0c835b29a011f5143',
                'VolumeSize': 200,
                'VolumeType': 'gp3'
            }
        }
    ],
    'NetworkInterfaces': [
        {
            'SubnetId': 'subnet-0e8b8c23ddb1ada49',
            'AssociatePublicIpAddress': True,
            'DeviceIndex': 0,
            'Groups': [
                'sg-0004eeb822745ac47'
            ]
        }
    ],
    'InstanceMarketOptions': {
        'MarketType': 'spot'
    },
    'MetadataOptions': {
        'HttpEndpoint': 'enabled',
        'HttpPutResponseHopLimit': 2,
        'HttpTokens': 'required'
    },
    'PrivateDnsNameOptions': {
        'HostnameType': 'ip-name',
        'EnableResourceNameDnsARecord': False,
        'EnableResourceNameDnsAAAARecord': False
    }
}
def test(i):
    start = time.time()
    # Launch the EC2 instance
    response = client.run_instances(**instance_params)
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
    