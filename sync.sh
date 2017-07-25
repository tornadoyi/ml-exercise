#!/usr/bin/expect -f
spawn scp -r ./ guyi@192.168.8.103:/home/guyi/Projects/ml-exercise
expect "*password:"
send "123456\r"

expect eof