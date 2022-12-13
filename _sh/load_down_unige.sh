#!/usr/bin/env bash
# just scp the results and logs file from daint back to the machine for analysis
# scp -r kermit@130.223.173.241:theta_project/theta_code/down.tar.gz ~/Dropbox/phd/Projects/theta_project/theta_code/
scp -r ~/.ssh/ pourmoha@login1.yggdrasil.hpc.unige.ch:./scratch/leave_one_out/res.tar.gz ./



