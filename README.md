$ sudo apt-get update
$ sudo apt-get install python-pip python-dev git
$ git clone https://github.com/CheeHau86/saeOcrDjango
$ sudo apt-get install docker
$ sudo pip install virtualenv
$ cd saeOcrDjango 
$ sudo apt-get update
$ source saeOcrDjango/bin/activate
$ docker build -t ocr-docker:0.0.1 .
$ docker run -p 8080:8080 ocr-docker:0.0.1
