# This script allows to run Jobs in Slurm with Perl

It also allows the following:

- Run your sbatch-scripts from a Perl-Environment
- Check for the parameters' sanity
- Run nvidia-smi on all nodes you requested to see your GPU-usage and get it logged into an easy-to-read csv-file, almost automatically.
- Get open ports on all requested nodes without effort
- Load modules from Perl and use them in sub-sruns
- Have a debug-system and log mostly everything easily

# Also, did you know ...

... that you can use commands like

> #SBATCH --signal=B:USR1@120

in non-bash-scripts like this perl-script? It works exactly the same way. Slurm does not care about the interpreter, 
it only reads the file as text-file and searches for /^#SBATCH (.\*)/.
