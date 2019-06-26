provider "aws" {
         region = "us-east-1"
         profile = "${var.aws_profile}"
         shared_credentials_file = "~/.aws/credentials"
}

resource "aws_instance" "mlbox" {
        ami = "ami-00ffbd996ef2211e3"
        instance_type = "p2.xlarge"
        key_name = "${var.keyname}"
        vpc_security_group_ids = ["${var.backend_sg}"]
        subnet_id = "${var.backend_sn}"

        tags = {
             Name = "${var.project}GPU"
             Terraform = "true"
             Project = "${var.project}"
        }
}

resource "aws_eip" "default" {
  instance = "${aws_instance.mlbox.id}"
  vpc      = true

  tags = {
       Name = "${var.project}EIP"
       Terraform = "true"
       Project = "${var.project}"
  }
}

variable "aws_profile" {
         description = "The AWS profile to use from the credentials file"
         default = "personal"
}

variable "backend_sg" {
         description = "The security group to use"
         default = "sg-0debce85f677f6b87"
}

variable "backend_sn" {
         description = "The subnet to use"
         default = "subnet-01bd898cf21b27a64"
}

variable "keyname" {
         description = "The keypair to use"
         default = "tim-user"
}

variable "project" {
         description = "Project name to tag with"
         default = "DeepLearning"
}
