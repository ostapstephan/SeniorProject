run everything 
python test.py 

Below is stuff that goes into the above script


on pi (out file of - goes into std in so that we can pipe it)
raspivid -t 0 -n -w 640 -h 480 -fps 30 -o - | nc 10.0.0.2 5001
raspivid -t 0 -n -w 640 -h 480 -fps 30 -o - | nc 10.0.0.2 5002

on pc (note make sure reader is greater fps than writer)
nc -l 5001 | mplayer -fps 100 -cache 1024 -
nc -l 5002 | mplayer -fps 100 -cache 1024 -

for piping into python
nc -l 5001 | python watchStream.py 
nc -l 5002 | python watchStream.py 
 
or run to pipe nc to fifo0 and fifo1 
./r0.sh and ./r1.sh

start netcat on pi to desktop
raspivid -t 0 -w 640 -h 480 -fps 30 -o - | nc 10.0.0.2 5001
raspivid -t 0 -w 640 -h 480 -fps 30 -o - | nc 10.0.0.2 5002

view the other devices on an ip 
nmap 10.0.0.1/24


ssh into the eye pi
ssh pi@10.0.0.3 -p 6622

ssh pi@10.0.0.5

measure temp 
watch /opt/vc/bin/vcgencmd measure_temp

FFMPEG documentation
https://www.ffmpeg.org/ffmpeg.html

how i setup ssh keys
https://www.raspberrypi.org/documentation/remote-access/ssh/passwordless.md

ssh and start stream 
#smlstream.sh is the 480x320
ssh pi@10.0.0.3 -p 6622 ~/stream.sh
ssh pi@10.0.0.5 ~/stream.sh

ssh and start stream python 
sshPi1 = sp.Popen(['ssh', 'pi@10.0.0.5', '~/smlstream.sh'], stdout = sp.PIPE )

using multithreading to reduce latency
https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/

//rename
for x in $(ls | grep 4-); do mv $x 1${x:1:-1}g; done



