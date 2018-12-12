provider "aws" {
         region = "us-east-1"
         profile = "${var.aws_profile}"
         shared_credentials_file = "~/.aws/credentials"
}

resource "aws_instance" "mlbox" {
        ami = "ami-00ffbd996ef2211e3"
        instance_type = "p2.xlarge"
        security_groups = ["sg-09f9c070f925b6758"]

        tags {
             Name = "DeepLearningGPU"
        }
}


variable "aws_profile" {
         description = "The AWS profile to use from the credentials file"
         default = "personal"
}