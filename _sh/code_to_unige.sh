#!/usr/bin/env bash
scp *.py pourmoha@login1.yggdrasil.hpc.unige.ch:./scratch/leave_one_out/
scp rf/*.py pourmoha@login1.yggdrasil.hpc.unige.ch:./scratch/leave_one_out/rf
scp helpers/*.py pourmoha@login1.yggdrasil.hpc.unige.ch:./scratch/leave_one_out/helpers
scp _slurm/* pourmoha@login1.yggdrasil.hpc.unige.ch:./scratch/leave_one_out/

echo "code pushed:"
date | cut -d " " -f5 | cut -d ":" -f1-2