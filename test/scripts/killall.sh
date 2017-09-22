ps -U $USER | egrep -v "ssh|screen|sftp|tmux" | awk 'NR>1 {print $1}' | xargs -t kill
