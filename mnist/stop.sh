#!/bin/bash

name='dmnist'
kill_sh_path='./log/kill.sh'

if [ -f $kill_sh_path ]; then
    echo 'call kill.sh'
    sh $kill_sh_path
fi

tmux kill-session -t $name