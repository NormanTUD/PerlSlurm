#!/usr/bin/perl

#SBATCH --signal=B:USR1@120

use strict;
use warnings FATAL => 'all';
use diagnostics;

use POSIX qw(strftime);
use File::Path qw(make_path);
use Data::Dumper;
use Term::ANSIColor;
use Hash::Util qw(lock_keys);
use autodie;
use File::Basename;
use IO::Socket::INET;
use Carp;
use Cwd;

use constant {
	DEFAULT_MIN_RAND_GENERATOR => 2_048,
	DEFAULT_MAX_RAND_GENERATOR => 65_500
};

use lib './perllib';
use Env::Modify;


my $received_exit_signal = 0;

sub debug (@);
sub debug_sub (@);
sub error (@);
sub debug_system ($);
sub warning (@);
sub dryrun (@);
sub message (@);
sub ok (@);
sub ok_debug (@);

sub usr1_signal {
	debug 'usr1_signal()';
	$received_exit_signal = 1;
	shutdown_script();
}

sub shutdown_script () {
	debug 'shutdown_script()';
	# Fill me with useful stuff that get's done once the job ends or is cancelled by anything not SIGKILL
}

# Add default values here so that they get used when none are specified
my %default_values = (
	number_of_allocated_gpus => 0,
	sleep_nvidia_smi => 30,
	sleep_db_up => 30
);
lock_keys(%default_values); # Lock keys so you don't do any typos later on

# Add options you want to your script here and in the analyze_args()-function.
my %options = (
	debug => 0,
	slurmid => $ENV{SLURM_JOB_ID},
	dryrun => 0,
	dryrunmsgs => 1,
	warnings => 1,
	messages => 1,
	sanitycheck => 1,
	number_of_allocated_gpus => $default_values{number_of_allocated_gpus},
	show_caller => 0,
	run_nvidia_smi => 1,
	sleep_nvidia_smi => $default_values{sleep_nvidia_smi},
	sleep_db_up => $default_values{sleep_db_up}
);
lock_keys(%options);

my %script_paths = (
	loggpu => get_working_directory()."/tools/loggpu.sh"
);

lock_keys(%script_paths);

get_environment_variables();
analyze_args(@ARGV);

debug_sub 'Outside of any function';
debug 'Locking keys of %options';
debug_options();

END {
	shutdown_script();
}

# Set up a signal handler for shutdown_script();
$SIG{USR1} = \&usr1_signal;
main();

# Run your main program here
sub main {
	debug_sub 'main()';

	cancel_message();
	sanity_check();

	module_load("MongoDB");

	run_nvidia_smi_periodically();

	wait_for_unfinished_jobs();
}

# If you have GPU-heavy jobs, e.g. training neural networks, and you want to see the GPU-usage of the job, you can use this function.
# To make sure it works, you have to write a file called
# /tmp/LOG_CUDA_VISIBLE_DEVICES
# on every node (created by the process you srun). It must only contain the contents of the shell-variable
# $CUDA_VISIBLE_DEVICES
# (e.g. via
# > echo $CUDA_VISIBLE_DEVICES > /tmp/LOG_CUDA_VISIBLE_DEVICES
# ). This is used to seperate GPUs you allocated and other GPUs on the machines so you only see the usage of your
# own processes.
#
# This requires that you have nvidia-GPUs and a working nvidia-smi installation. For every job-id, you'll get a single
# folder with nvidia-servername/gpu_usage.csv. This file will contain only one header and then the values of the GPUs
# every --sleep_nvidia_smi seconds.
sub run_nvidia_smi_periodically {
	debug_sub "run_nvidia_smi_periodically()";
	if(!$options{run_nvidia_smi}) {
		debug "Not running nvidia-smi because of --run_nvidia_smi=0";
		return 1;
	}

	if(exists $ENV{'SLURM_JOB_NODELIST'}) {
		debug "Environment variable `SLURM_JOB_NODELIST` exists ($ENV{SLURM_JOB_NODELIST}), therefore, I am running nvidia-smi periodically";

		debug "Forking for run_nvidia_smi_periodically()";
		my $pid = fork();
		error "ERROR Forking for nvidia-smi: $!" if not defined $pid;
		if (not $pid) {
			debug "Inside fork";
			my $slurm_nodes = $ENV{'SLURM_JOB_NODELIST'};
			my @server = get_servers_from_SLURM_NODES($slurm_nodes);

			while (!$received_exit_signal) {
				foreach my $this_server (@server) {
					my $nvidialogpath = get_working_directory()."/".$ENV{SLURM_JOB_ID}."/nvidia-$this_server/";
					if(!-d $nvidialogpath) {
						debug "$nvidialogpath does not exist yet, creating it";
						debug_system("mkdir -p $nvidialogpath") unless -d $nvidialogpath;
					}

					my $nvidialogfile = "$nvidialogpath/gpu_usage.csv";
					if(!-e $nvidialogfile) {
						debug_system("touch $nvidialogfile");
					}

					my $command = "bash $script_paths{loggpu} $nvidialogfile";
					my $sshcommand = "ssh -o LogLevel=ERROR $this_server '$command'";
					my $return_code = debug_system($sshcommand);
					if($return_code) {
						warning "$sshcommand seems to have failed! Exit-Code: $return_code";
					}
				}

				debug "Sleeping for $options{sleep_nvidia_smi} (set value via `... sbatch.pl --sleep_nvidia_smi=n ...`) seconds before ".
				"executing nvidia-smi on each server again";
				sleep $options{sleep_nvidia_smi};
			}
			exit(0);
		}
	} else {
		message "\$ENV{SLURM_JOB_NODELIST} not defined, are you sure you are in a Slurm-Job? Not running nvidia-smi."
	}
}

# Prints out a cancel-message showing how you can sanely cancel the process (and allowing it, e.g., to properly
# shut down a database.
sub cancel_message {
	if(defined $options{slurmid}) {
		message "If you want to cancel this job, please use\n".
		"\t\t\tscancel --signal=USR1 --batch $options{slurmid}\n".
		"\t\tThis way, the database can be shut down correctly.";
	}
}

# Use this if you forked something and want to wait for it after other processes in the main script ran.
sub wait_for_unfinished_jobs {
	debug_sub "wait_for_unfinished_jobs()";
	debug 'Waiting for started subjobs to exit';
	wait;
	ok_debug 'Done waiting for started subjobs';
}

# Use this to spawn subprocesses that run on different nodes than the main node.
# > run_srun("python3 ...");
sub run_srun {
	my $run = shift;
	debug_sub "run_srun($run)";

	my $gpu_options = '';
	if(defined $options{number_of_allocated_gpus}) {
		$gpu_options = ' --gres=gpu:1 ';
		$gpu_options .= ' --accel-bind=g ';
		$gpu_options .= ' --exclusive ';
		$gpu_options =~ s#\s+# #g;
		debug "Set \$gpu_options to `$gpu_options`";
	}

	### Important! Add something like "-n 10" after the srun to spawn 10 sruns.
	my $command = qq#srun $gpu_options --mpi=none --no-kill bash -c "$run"#;
	$command =~ s#\s{2,}# #g;

	debug $command;

	if(!$options{dryrun}) {
		fork_and_dont_wait($command);
	} else {
		dryrun "Not running\n\t\t$command\n\t\tbecause of --dryrun";
	}
}

# Fork, run a command and do not wait for it to return.
sub fork_and_dont_wait {
	my $command = shift;
	debug_sub "fork_and_dont_wait($command)";

	if(!$options{dryrun}) {
		my $pid = fork();
		error 'SOMETHING WENT HORRIBLY WRONG WHILE FORKING!' if not defined $pid;
		if (not $pid) {
			debug "In child process $$";
			exec($command) or error "Something went wrong executing '$command': $!";;
		} else {
			debug 'In Parent process';
		}
	} else {
		dryrun "Not really forking_and_not_waiting for $command, because of --dryrun";
	}
}

# Use this to create a fork and run a sub-references inside of it.
sub fork_sub_call {
	my ($subref, @args) = @_;

	if(ref $subref ne 'CODE') {
		error 'Wrong var-type. Should be subref, nothing else.';
	} else {
		my $pid = fork();
		error 'SOMETHING WENT WRONG WHILE FORKING!' if not defined $pid;

		if (not $pid) {
			return &{$subref}->(@args);
		}
	}
	wait();
}

# Do your sanity checks here. It is automatically skipped it you called this script with --nosanitycheck.
sub sanity_check {
	debug_sub 'sanity_check()';

	if(!$options{sanitycheck}) {
		message 'Disabled sanity check';
	} else {
		if(!$options{slurmid}) {
			warning 'No Slurm-ID!';
		}

		# Don't forget to add sanity checks for your parameters!
	}
}

# Outputs a help.
sub _help {
	debug_sub '_help()';

	my $cli_code = color('yellow bold');
	my $reset = color('reset');

	print <<"EOF";

=== HELP ===

Write your help-document here!

Parameters:
    --help                                  This help
    --nosanitycheck                         Disables the sanity checks (not recommended!)
    --run_nvidia_smi=[01]                   Run (1) or don't run (0) nvidia-smi periodically
					    to get information about the GPU usage of the workers
    --sleep_nvidia_smi=n                    Sleep n seconds after each try to get 
					    nvidia-smi-gpu-infos (default: $default_values{sleep_nvidia_smi} seconds)

Debug and output parameters:
    --nomsgs                                Disables messages
    --debug                                 Enables lots and lots of debug outputs
    --dryrun                                Run this script without any side effects
					    (i.e. not creating files, not starting
					    workers etc.)
    --nowarnings                            Disables the outputting of warnings 
					    (not recommended!)
    --nodryrunmsgs                          Disables the outputting of --dryrun-messages
    --run_tests                             Runs a bunch of tests and exits without doing 
					    anything else
    --show_caller                           Show caller in all kinds of messages (useful for
					    debugging)
EOF
}

sub analyze_args {
	my @args = @_;

	foreach my $arg (@args) {
		if ($arg eq '--dryrun') {
			$options{dryrun} = 1;
		} elsif ($arg eq '--nomsgs') {
			$options{messages} = 0;
		} elsif ($arg eq '--nowarnings') {
			$options{warnings} = 0;
		} elsif ($arg eq '--nosanitycheck') {
			$options{sanitycheck} = 0;
		} elsif ($arg eq '--nodryrunmsgs') {
			$options{dryrunmsgs} = 0;
		} elsif ($arg eq '--show_caller') {
			$options{show_caller} = 1;
		} elsif ($arg eq '--debug') {
			$options{debug} = 1;
		} elsif ($arg =~ m#^--run_nvidia_smi=([01])$#) {
			$options{run_nvidia_smi} = $1;
		} elsif ($arg =~ m#^--sleep_nvidia_smi=(\d+)$#) {
			$options{sleep_nvidia_smi} = $1;
		} elsif ($arg eq '--help') {
			_help();
			exit(0);
		} else {
			print color("red underline")."Unknown command line switch: $arg".color("reset")."\n";
			_help();
			exit(1);
		}
	}
}

# Use this as qx-replacement. It enables debugging of the executed command.
sub debug_qx ($) {
	my $command = shift;
	debug_sub "debug_qx($command)";

	return qx($command);
}

# Use this to run system-calls. It enables debugging-outputs with --debug and returns the 
# original (unshifted) exit-code.
sub debug_system ($) {
	my $command = shift;
	debug_sub "debug_system($command)";

	system($command);
	my $exit_code = $? >> 8;
	debug "EXIT-Code: $exit_code";
	return $exit_code;
}

# Get infos about allocated GPU-devices (if any).
sub _get_gpu_info {
	debug_sub '_get_gpu_info()';
	my $gpu_device_ordinal = _get_environment_variable('GPU_DEVICE_ORDINAL');
	if(defined $gpu_device_ordinal && $gpu_device_ordinal !~ /dev/i) {
		my @gpus = split(/,/, $gpu_device_ordinal);
		return scalar @gpus;
	} else {
		message 'No GPUs allocated';
		return $default_values{number_of_allocated_gpus};
	}
}

# Get some information about the loaded environment, e.g. the number of allocated GPUs per Node.
# Add more information here if you need to.
sub get_environment_variables {
	debug_sub 'get_environment_variables()';

	$options{number_of_allocated_gpus} = _get_gpu_info();

	return 1;
}

# If available, return the value of a Shell-Variable.
sub _get_environment_variable {
	my $name = shift;
	debug_sub "_get_environment_variable($name)";
	return $ENV{$name};
}

# Load an array of modules.
sub modules_load {
	my @modules = @_;
	debug_sub 'modules_load('.join(', ', @modules).')';
	foreach my $mod (@modules) {
		module_load($mod);
	}

	return 1;
}

# Use this to load a module like you would in a bash-script with LMOD. It will be available for all
# subsequent system-calls.
sub module_load {
	my $toload = shift;
	debug_sub "module_load($toload)";

	if($toload) {
		if($options{dryrun}) {
			dryrun "Not loading module $toload because of --dryrun";
		} else {
			my $lmod_path = $ENV{LMOD_CMD};
			if($lmod_path) {
				my $command = "eval \$($lmod_path sh load $toload)";
				debug $command;
				local $Env::Modify::CMDOPT{startup} = 1;
				Env::Modify::system($command);
			} else {
				warning "LMOD_CMD not set! Cannot load modules.";
			}
		}
	} else {
		warning 'Empty module_load!';
	}
	return 1;
}

# Used for debugging the options of this sbatch.pl-call.
sub debug_options {
	debug_sub 'debug_options()';
	debug 'This script file: '.get_working_directory()."/".__FILE__;
	debug split(/\R/, Dumper \%options);
	return 1;
}

# Checks whether a program is available, e.g. in the $PATH of the shell.
# Returns 1 if it is and 0 if not.
sub program_installed {
	my $program = shift;
	debug_sub "program_installed($program)";
	my $ret = qx(whereis $program | sed -e 's/^$program: //');
	chomp $ret;
	my @paths = split /\s*/, $ret;
	my $exists = 0;
	PATHS: foreach my $this_file (@paths) {
		if(-e $this_file) {
			$exists = 1;
			last PATHS;
		}
	}

	if($exists) {
		debug "$program already installed";
	} else {
		warning "$program does not seem to be installed. Please install it!";
	}

	return $exists;
}

# Gets the IP-address of the node this script runs on.
# Source:
# https://stackoverflow.com/questions/330458/how-can-i-determine-the-local-machines-ip-addresses-from-perl
sub get_local_ip_address {
	debug_sub 'get_local_ip_address()';
	my $socket = IO::Socket::INET->new(
		Proto       => 'udp',
		PeerAddr    => '198.41.0.4', # a.root-servers.net
		PeerPort    => '53', # DNS
	);

	# A side-effect of making a socket connection is that our IP address
	# is available from the 'sockhost' method
	my $local_ip_address = $socket->sockhost;

	return $local_ip_address;
}

# Check if a port on a single server is open.
# > if(server_port_is_open($server, $port)) { print "Port $port is open on Server $server"; }
sub server_port_is_open {
	my $server = shift;
	my $port = shift;

	debug_sub "server_port_is_open($server, $port)";

	local $| = 1;

	# versucht ein TCP-Socket auf dem Server mit dem Port zu öffnen; wenn das geht, ist der Port nicht offen (return 0)

	my $socket = IO::Socket::INET->new(
		PeerHost => $server,
		PeerPort => $port,
		Proto => 'tcp'
	);

	if($socket) {
		return 0;
	} else {
		return 1;
	}
}

# Extracts Information about the allocated servers from the SLURM_JOB_NODELIST and puts them into
# a convienent array.
sub get_servers_from_SLURM_NODES {
	my $string = shift;
	debug_sub "get_servers_from_SLURM_NODES($string)";
	my @server;
	while ($string =~ m#(.*?)\[(.*?)\](?:,|\R|$)#gi) {
		my $servercluster = $1;
		my $servernumbers = $2;
		foreach my $thisservernumber (split(/,/, $servernumbers)) {
			if($servernumbers !~ /-/) {
				push @server, "$servercluster$thisservernumber";
			}
		}

		if($servernumbers =~ m#(\d+)-(\d+)#) {
			push @server, map { "$servercluster$_" } $1 .. $2;
		}
	}

	if(@server) {
		return @server;
	} else {
		return ('127.0.0.1');
	}
}

# Returns a random number between DEFAULT_MIN_RAND_GENERATOR and DEFAULT_MAX_RAND_GENERATOR,
# unless otherwise specified (e.g. get_random_number(10, 100)).
sub get_random_number {
	my $minimum = shift // DEFAULT_MIN_RAND_GENERATOR;
	my $maximum = shift // DEFAULT_MAX_RAND_GENERATOR;
	debug_sub "get_random_number($minimum, $maximum)";
	my $x = $minimum + int(rand($maximum - $minimum));
	debug "random_number -> $x";
	return $x;
}

# Use this if you want to know whether a port is open on all servers you supplied as
# second argument.
sub test_port_on_multiple_servers {
	my ($port, @servers) = @_;
	debug_sub "test_port_on_multiple_servers($port, (".join(', ', @servers).'))';
	my $is_open_everywhere = 1;
	THISFOREACH: foreach my $server (@servers) {
		if(!server_port_is_open($server, $port)) {
			$is_open_everywhere = 0;
			print "Port $port was not open on server $server";
			last THISFOREACH;
		}
	}
	if($is_open_everywhere) {
		ok_debug "Port $port is open everywhere!";
	} else {
		warning "Port $port is NOT open everywhere";
	}
	return $is_open_everywhere;
}

# Returns a port number that is open on all requested nodes
sub get_open_ports {
	debug_sub 'get_open_ports()';

	my $slurm_nodes = '127.0.0.1';
	if(exists $ENV{'SLURM_JOB_NODELIST'}) {
		$slurm_nodes = $ENV{'SLURM_JOB_NODELIST'};
	}

	debug "Slurm-Nodes: $slurm_nodes";

	my @server = get_servers_from_SLURM_NODES($slurm_nodes);
	my $port = get_random_number();
	while (!test_port_on_multiple_servers($port, @server)) {
		$port = get_random_number();
	}

	ok_debug "Port: $port";

	return $port;
}

# Since slurm copies the main script to somewhere else, you might use this to get the original working directory.
# If you're not in a Slurm-Job, it will return the normal CWD, otherwise, it will ask scontrol where your original
# working directory was and return this.
sub get_working_directory {
	debug_sub "get_working_directory()";
	my $cwd = '';
	if(exists $ENV{SLURM_JOB_ID}) {
		my $command = qq#scontrol show job $ENV{SLURM_JOB_ID} | egrep "^\\s*WorkDir=" | sed -e 's/^\\s*WorkDir=//'#;
		$cwd = debug_qx($command);
		chomp $cwd;
	} else {
		debug "Calling getcwd()";
		$cwd = getcwd();
	}

	if(!defined $cwd) {
		error "WARNING: CWD Seems to be empty!"
	}

	if(!-d $cwd) {
		error "WARNING: $cwd could not be found!!!";
	}

	return $cwd;
}

# Use this to run a command on all allocated nodes.
# E.g. 
# > run_ssh_command_on_all_nodes("touch /abc");
# This only works when your slurm is configured so that you can access the requested nodes via SSH from
# your username.
sub run_ssh_command_on_all_nodes {
	my $command = shift;
	debug_sub "run_ssh_command_on_all_nodes($command)";

	if(exists $ENV{SLURM_JOB_ID} && exists $ENV{SLURM_JOB_NODELIST}) {
		my @server = get_servers_from_SLURM_NODES($ENV{SLURM_JOB_NODELIST});

		foreach my $this_server (@server) {
			debug_system("ssh -o LogLevel=ERROR $this_server '$command'");
		}
	} else {
		debug "Not in a Slurm-Job, therefore, no allocated nodes.";
	}
}

#### Debug-Outputs

# Use this to implement new methods for creating outputs. This way, it will be easy to implement a redirect to 
# a log file for example.
# > stderr_debug_log $show_this_message, $message_itself;
sub stderr_debug_log ($$) {
	my $show = shift;
	my $string = shift;

	return unless $show;
	warn $string;
}

# Use this for messages if you implement a dry-run-mode in your program that acts as if it did something,
# but actually does nothing. 
# This is very useful to test this script itself.
# > dryrun "This would have ran if you didn't disable it with --dryrun";
sub dryrun (@) {
	my @msgs = @_;
	if($options{dryrun}) {
		foreach my $msg (@msgs) {
			stderr_debug_log $options{dryrunmsgs}, color('magenta')."DRYRUN:\t\t$msg".color('reset')."\n";
		}

		return 1;
	}
}

# Use this for every "OK" that is not neccessary to see in non-debug-mode.
# > ok_debug "Ok, debugging this worked";
sub ok_debug (@) {
	foreach (@_) {
		stderr_debug_log $options{debug}, color('green')."OK_DEBUG:\t$_".color('reset')."\n";
	}
}

# Use this for everything that went Ok so you can be happy if your whole console lights up green.
# > ok "Ok Message";
# It will get printed to stdout
sub ok (@) {
	foreach (@_) {
		stderr_debug_log 1, color('green')."OK:\t\t$_".color('reset')."\n";
	}
}

# Use this as
# > message "Message";
# It will print messages to stdout with a special color.
sub message (@) {
	foreach (@_) {
		stderr_debug_log $options{messages}, color('cyan')."MESSAGE: \t$_".color('reset')."\n";
	}
}

# Use this as
# > warning "Warning message";
# for warning-messages.
# It will get printed to stderr in a special color.
sub warning (@) {
	#return if !$options{warnings};
	my @warnings = @_;
	my $sub_name = '';
	if($options{show_caller}) {
		$sub_name = (caller(1))[3];
		$sub_name = " (from $sub_name)";
	}
	foreach my $wrn (@warnings) {
		stderr_debug_log $options{warnings}, color('yellow')."WARNING$sub_name:\t$wrn".color('reset')."\n";
	}
}

# Use this with
# > debug "Debug-message";
# so you can enable or disable this output (very useful for debugging)
sub debug (@) {
	foreach (@_) {
		stderr_debug_log $options{debug}, color('cyan')."DEBUG:\t\t$_".color('reset')."\n";
	}
}

# Use this with 
# > debug_sub("sub_name(...)");
# at the beginning of a function, so that you can easily trace your functions with the --debug flag

sub debug_sub (@) {
	foreach (@_) {
		stderr_debug_log $options{debug}, color('bold cyan')."DEBUG_SUB:\t$_".color('reset')."\n";
	}
}

# Use this with
# > error "Errormessage";
# Result: The program dies and it prints out the message to stderr
sub error (@) {
	foreach (@_) {
		stderr_debug_log 1, color('red')."ERROR:\t\t$_.".color('reset')."\n";
	}
	exit 1;
}
