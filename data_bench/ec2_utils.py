import boto3

client = boto3.Session().client('ec2')

def get_block_device_mappings(params, size_gb=None):
    response = client.describe_images(ImageIds=[params["ImageId"]])
    blocking_device_mapping = response["Images"][0]["BlockDeviceMappings"]
    if size_gb:
        blocking_device_mapping[0]["Ebs"]["VolumeSize"] = size_gb
    return blocking_device_mapping

def get_subnet_id(availability_zone: str) -> str:
    response = client.describe_subnets()
    subnet_ids = [subnet["SubnetId"] for subnet in response["Subnets"] if subnet["AvailabilityZone"] == availability_zone]
    assert subnet_ids, f"No subnets found for availability zone {availability_zone}"
    return subnet_ids[0]