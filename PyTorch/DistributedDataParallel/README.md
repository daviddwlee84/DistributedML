```bash
# TODO: 
$ python single_forward_backward.py

$ python basic_use_case.py
Running basic DDP example on rank 3.
Running basic DDP example on rank 0.
Running basic DDP example on rank 2.
Running basic DDP example on rank 1.
Running DDP checkpoint example on rank 0.
Running DDP checkpoint example on rank 1.
Running DDP checkpoint example on rank 3.
Running DDP checkpoint example on rank 2.
Running DDP with model parallel example on rank 1.
Running DDP with model parallel example on rank 0.
```

Single-node multi-worker

```bash
$ torchrun --standalone --nnodes=1 --nproc-per-node=2 `pwd`/elastic_ddp.py
W0813 20:04:44.287418 139760121915136 torch/distributed/run.py:779]
W0813 20:04:44.287418 139760121915136 torch/distributed/run.py:779] *****************************************
W0813 20:04:44.287418 139760121915136 torch/distributed/run.py:779] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
W0813 20:04:44.287418 139760121915136 torch/distributed/run.py:779] *****************************************
Start running basic DDP example on rank 1.
Start running basic DDP example on rank 0.
```

Multi-node multi-worker

> 1. Two nodes, each have 2 GPUs
> 2. Their networks are fine and can communicate with each other
>    1. Test basic connectivity with `ping` to others' IP or hostname
>    2. Test port by run this on master `nc -l 29400` and run this on worker `nc -zv $MASTER_ADDR 29400`
> 3. rdzv (Rendezvous) tends to use hostname, make sure hostname is recognized by DNS, otherwise edit `/etc/hosts`

```bash
# On Master Node
$ torchrun --nnodes=2 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=localhost:29400 --node_rank=0 `pwd`/elastic_ddp.py
W0813 22:22:54.570086 140209818330880 torch/distributed/run.py:779]
W0813 22:22:54.570086 140209818330880 torch/distributed/run.py:779] *****************************************
W0813 22:22:54.570086 140209818330880 torch/distributed/run.py:779] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
W0813 22:22:54.570086 140209818330880 torch/distributed/run.py:779] *****************************************
Start running basic DDP example on rank 1.
Start running basic DDP example on rank 0.
W0813 22:23:32.979031 140204091807488 torch/distributed/elastic/rendezvous/dynamic_rendezvous.py:1267] The node 'xxxxx' has failed to send a keep-alive heartbeat to the rendezvous '100' due to an error of type RendezvousTimeoutError.

# On Worker Node
export MASTER_ADDR=...
export MASTER_PORT=29400

$ torchrun --nnodes=2 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT --node_rank=1 `pwd`/elastic_ddp.py
W0813 22:23:01.756441 139918533302016 torch/distributed/run.py:779]
W0813 22:23:01.756441 139918533302016 torch/distributed/run.py:779] *****************************************
W0813 22:23:01.756441 139918533302016 torch/distributed/run.py:779] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
W0813 22:23:01.756441 139918533302016 torch/distributed/run.py:779] *****************************************
Start running basic DDP example on rank 3.
Start running basic DDP example on rank 2.
```

Multi-node multi-worker - CNN


```bash
# Run it once, assume this folder is shared among machines and have same route
python ../download_mnist.py

# On Master Node
$ torchrun --nnodes=2 --nproc_per_node=1 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=l
ocalhost:29400 --node_rank=0 `pwd`/elastic_ddp_cnn.py
Initialized process group with rank 0.
Running on rank 0. (local device: 0)
Epoch [0/10], Batch [0/469], Loss: 2.3098
Epoch [0/10], Batch [100/469], Loss: 0.0644
Epoch [0/10], Batch [200/469], Loss: 0.1184
Epoch [0/10], Batch [300/469], Loss: 0.0800
Epoch [0/10], Batch [400/469], Loss: 0.0355
Epoch [0/10] completed. Loss: 0.1915
...

# On Worker Node
$ torchrun --nnodes=2 --nproc_per_node=1 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT --node_rank=1 `pwd`/elastic_ddp_cnn.py                                                                                                                             ^Initialized process group with rank 1.
Running on rank 1. (local device: 0)
```

---

- [Rendezvous — PyTorch 2.4 documentation](https://pytorch.org/docs/stable/elastic/rendezvous.html)
- [Error Propagation — PyTorch 2.4 documentation](https://pytorch.org/docs/stable/elastic/errors.html)
- [How to use `MASTER_ADDR` in a distributed training script · Issue #65992 · pytorch/pytorch](https://github.com/pytorch/pytorch/issues/65992)

```bash
$ torchrun -h
usage: torchrun [-h] [--nnodes NNODES] [--nproc_per_node NPROC_PER_NODE] [--rdzv_backend RDZV_BACKEND]
                [--rdzv_endpoint RDZV_ENDPOINT] [--rdzv_id RDZV_ID] [--rdzv_conf RDZV_CONF] [--standalone]
                [--max_restarts MAX_RESTARTS] [--monitor_interval MONITOR_INTERVAL]
                [--start_method {spawn,fork,forkserver}] [--role ROLE] [-m] [--no_python] [--run_path]
                [--log_dir LOG_DIR] [-r REDIRECTS] [-t TEE] [--node_rank NODE_RANK] [--master_addr MASTER_ADDR]
                [--master_port MASTER_PORT]
                training_script ...

Torch Distributed Elastic Training Launcher

positional arguments:
  training_script       Full path to the (single GPU) training program/script to be launched in parallel, followed by
                        all the arguments for the training script.
  training_script_args

optional arguments:
  -h, --help            show this help message and exit
  --nnodes NNODES       Number of nodes, or the range of nodes in form <minimum_nodes>:<maximum_nodes>.
  --nproc_per_node NPROC_PER_NODE
                        Number of workers per node; supported values: [auto, cpu, gpu, int].
  --rdzv_backend RDZV_BACKEND
                        Rendezvous backend.
  --rdzv_endpoint RDZV_ENDPOINT
                        Rendezvous backend endpoint; usually in form <host>:<port>.
  --rdzv_id RDZV_ID     User-defined group id.
  --rdzv_conf RDZV_CONF
                        Additional rendezvous configuration (<key1>=<value1>,<key2>=<value2>,...).
  --standalone          Start a local standalone rendezvous backend that is represented by a C10d TCP store on port
                        29400. Useful when launching single-node, multi-worker job. If specified --rdzv_backend,
                        --rdzv_endpoint, --rdzv_id are auto-assigned; any explicitly set values are ignored.
  --max_restarts MAX_RESTARTS
                        Maximum number of worker group restarts before failing.
  --monitor_interval MONITOR_INTERVAL
                        Interval, in seconds, to monitor the state of workers.
  --start_method {spawn,fork,forkserver}
                        Multiprocessing start method to use when creating workers.
  --role ROLE           User-defined role for the workers.
  -m, --module          Change each process to interpret the launch script as a Python module, executing with the same
                        behavior as 'python -m'.
  --no_python           Skip prepending the training script with 'python' - just execute it directly. Useful when the
                        script is not a Python script.
  --run_path            Run the training script with runpy.run_path in the same interpreter. Script must be provided
                        as an abs path (e.g. /abs/path/script.py). Takes precedence over --no_python.
  --log_dir LOG_DIR     Base directory to use for log files (e.g. /var/log/torch/elastic). The same directory is re-
                        used for multiple runs (a unique job-level sub-directory is created with rdzv_id as the
                        prefix).
  -r REDIRECTS, --redirects REDIRECTS
                        Redirect std streams into a log file in the log directory (e.g. [-r 3] redirects both
                        stdout+stderr for all workers, [-r 0:1,1:2] redirects stdout for local rank 0 and stderr for
                        local rank 1).
  -t TEE, --tee TEE     Tee std streams into a log file and also to console (see --redirects for format).
  --node_rank NODE_RANK
                        Rank of the node for multi-node distributed training.
  --master_addr MASTER_ADDR
                        Address of the master node (rank 0). It should be either the IP address or the hostname of
                        rank 0. For single node multi-proc training the --master_addr can simply be 127.0.0.1; IPv6
                        should have the pattern `[0:0:0:0:0:0:0:1]`.
  --master_port MASTER_PORT
                        Port on the master node (rank 0) to be used for communication during distributed training.
```
