$ sudo apt-get update
$ sudo apt-get install python3-pip -y
$ sudo apt-get install python-pip python-dev git -y
$ sudo pip3 install virtualenv
$ git clone https://github.com/CheeHau86/saeOcrDjango.git
$ cd sae saeOcrDjango 
$ virtualenv ocrenv
$ source ocrenv/bin/activate
$ sudo apt-get install python3-opencv -y
$ pip install -r requirements.txt 
$ sudo apt install tesseract-ocr -y
$ sudo apt-get update

#log out and
#the model file is 370+Mb, unable to upload to github. we need to manually upload into server 
#the model file is share in google drive https://drive.google.com/drive/u/0/folders/17PW5L1QGKhc_kC7Vq6VimDIXD0UXcTnb

$ scp -i "key.pem" localDesktopLocation/model.ckpt-150491.data-00000-of-00001 ubuntu@ipAddress:/home/ubuntu/ocrDocker/models/model_150ksteps

$ cd saeOcrDjango 
$ source ocrenv/bin/activate
$ python manage.py runserver 0.0.0.0:8088
