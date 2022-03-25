2022-03-24 16:19:14 INFO: Create database interface took 0:00:00.099664 
2022-03-24 16:19:14 INFO: ** Notifying external sources that the database is ready for reading **
2022-03-24 16:19:14 INFO: 0.0.0.0:14001 Reading database
2022-03-24 16:19:14 INFO: database is at /home/juan/CLSpiNN/experiments/closed_loop/reports/2022-03-24-16-18-49-779357/run_1/input_output_database.sqlite3
2022-03-24 16:19:14 INFO: 0.0.0.0:33974 Reading database
2022-03-24 16:19:14 INFO: database is at /home/juan/CLSpiNN/experiments/closed_loop/reports/2022-03-24-16-18-49-779357/run_1/input_output_database.sqlite3
2022-03-24 16:19:14 INFO: Notifying the toolchain that the database has been read
2022-03-24 16:19:14 INFO: Waiting for message to indicate that the simulation has started or resumed
2022-03-24 16:19:14 INFO: Create notification protocol took 0:00:00.000523 
Waiting for cores to be either in PAUSED or READY state
|0%                          50%                         100%|
 2022-03-24 16:19:14 INFO: Listening for traffic from motor_neurons on board 172.16.223.29 on 0.0.0.0:39317
2022-03-24 16:19:14 INFO: Notifying the toolchain that the database has been read
2022-03-24 16:19:14 INFO: Waiting for message to indicate that the simulation has started or resumed
2022-03-24 16:19:14 INFO: ** Confirmation from 127.0.0.1:14001 received, continuing **
2022-03-24 16:19:14 INFO: ** Confirmation from 127.0.0.1:33974 received, continuing **
2022-03-24 16:19:14 INFO: global chip 4, 4 on 172.16.223.29 is chip 4, 4 on 172.16.223.29
2022-03-24 16:19:14 INFO: Runtime Update exited with SpiNNManCoresNotInStateException after 0:00:00.051967
2022-03-24 16:19:14 ERROR: An error has occurred during simulation
2022-03-24 16:19:14 ERROR: waiting for cores odict_keys([(4, 4, 16), (4, 4, 15)]) to reach one of [<CPUState.PAUSED: 10>, <CPUState.READY: 5>]
2022-03-24 16:19:14 INFO: 

Attempting to extract data


Getting Router Provenance
|0%                          50%                         100%|
 ============================================================
2022-03-24 16:19:14 ERROR: 4, 4, 16: RUN_TIME_EXCEPTION (API) IF_curr_exp_con
2022-03-24 16:19:14 ERROR: r0=0x00000013 r1=0x00000000 r2=0x00004000 r3=0x00000934
2022-03-24 16:19:14 ERROR: r4=0x60243338 r5=0x00400014 r6=0x004004B8 r7=0x00000000
2022-03-24 16:19:14 ERROR: PSR=0x6000001F SR=0x0040FC50 LR=0x000005BC
2022-03-24 16:19:14 ERROR: 4, 4, 15: RUN_TIME_EXCEPTION (API) IF_curr_exp_con
2022-03-24 16:19:14 ERROR: r0=0x00000013 r1=0x00000000 r2=0x00004000 r3=0x00000934
2022-03-24 16:19:14 ERROR: r4=0x6025F0E4 r5=0x00400014 r6=0x004004B8 r7=0x00000000
2022-03-24 16:19:14 ERROR: PSR=0x6000001F SR=0x0040FC50 LR=0x000005BC
Extracting IOBUF from the machine
|0%                          50%                         100%|
 ============================================================
2022-03-24 16:19:14 ERROR: 4, 4, 16: Unable to allocate synapse parameters array - Out of DTCM (neuron_impl_standard.h: 145)
2022-03-24 16:19:14 ERROR: 4, 4, 15: Unable to allocate synapse parameters array - Out of DTCM (neuron_impl_standard.h: 145)
Traceback (most recent call last):
  File "integrator.py", line 730, in <module>
    run_spinnaker_sim()
  File "integrator.py", line 372, in run_spinnaker_sim
    p.run(RUN_TIME)
  File "/home/rowleya/SpiNNaker/sPyNNaker/spynnaker8/__init__.py", line 675, in run
    return __pynn["run"](simtime, callbacks=callbacks)
  File "/home/rowleya/SpiNNaker/lib/python3.8/site-packages/PyNN-0.9.6-py3.8.egg/pyNN/common/control.py", line 111, in run
    return run_until(simulator.state.t + simtime, callbacks)
  File "/home/rowleya/SpiNNaker/lib/python3.8/site-packages/PyNN-0.9.6-py3.8.egg/pyNN/common/control.py", line 93, in run_until
    simulator.state.run_until(time_point)
  File "/home/rowleya/SpiNNaker/sPyNNaker/spynnaker8/spinnaker.py", line 88, in run_until
    self._run_wait(tstop - self.t)
  File "/home/rowleya/SpiNNaker/sPyNNaker/spynnaker8/spinnaker.py", line 119, in _run_wait
    super(SpiNNaker, self).run(duration_ms, sync_time)
  File "/home/rowleya/SpiNNaker/sPyNNaker/spynnaker/pyNN/abstract_spinnaker_common.py", line 281, in run
    super().run(run_time, sync_time)
  File "/home/rowleya/SpiNNaker/SpiNNFrontEndCommon/spinn_front_end_common/interface/abstract_spinnaker_base.py", line 819, in run
    self._run(run_time, sync_time)
  File "/home/rowleya/SpiNNaker/SpiNNFrontEndCommon/spinn_front_end_common/interface/abstract_spinnaker_base.py", line 1026, in _run
    self._do_run(step, graph_changed, n_sync_steps)
  File "/home/rowleya/SpiNNaker/SpiNNFrontEndCommon/spinn_front_end_common/interface/abstract_spinnaker_base.py", line 3061, in _do_run
    raise run_e
  File "/home/rowleya/SpiNNaker/SpiNNFrontEndCommon/spinn_front_end_common/interface/abstract_spinnaker_base.py", line 3042, in _do_run
    self.__do_run(
  File "/home/rowleya/SpiNNaker/SpiNNFrontEndCommon/spinn_front_end_common/interface/abstract_spinnaker_base.py", line 3021, in __do_run
    self._execute_runtime_update(n_sync_steps)
  File "/home/rowleya/SpiNNaker/SpiNNFrontEndCommon/spinn_front_end_common/interface/abstract_spinnaker_base.py", line 2889, in _execute_runtime_update
    chip_runtime_updater(
  File "/home/rowleya/SpiNNaker/SpiNNFrontEndCommon/spinn_front_end_common/interface/interface_functions/chip_runtime_updater.py", line 42, in chip_runtime_updater
    txrx.wait_for_cores_to_be_in_state(
  File "/home/rowleya/SpiNNaker/SpiNNMan/spinnman/transceiver.py", line 1998, in wait_for_cores_to_be_in_state
    raise SpiNNManCoresNotInStateException(
spinnman.exceptions.SpiNNManCoresNotInStateException: waiting for cores odict_keys([(4, 4, 16), (4, 4, 15)]) to reach one of [<CPUState.PAUSED: 10>, <CPUState.READY: 5>]
Process Process-3:
Traceback (most recent call last):
  File "/usr/lib/python3.8/multiprocessing/process.py", line 315, in _bootstrap
    self.run()
  File "/usr/lib/python3.8/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "integrator.py", line 558, in get_outputs
    if end_of_sim.value == 1:
  File "/usr/lib/python3.8/multiprocessing/managers.py", line 1154, in get
    return self._callmethod('get')
  File "/usr/lib/python3.8/multiprocessing/managers.py", line 835, in _callmethod
    kind, result = conn.recv()
  File "/usr/lib/python3.8/multiprocessing/connection.py", line 250, in recv
    buf = self._recv_bytes()
  File "/usr/lib/python3.8/multiprocessing/connection.py", line 414, in _recv_bytes
    buf = self._recv(4)
  File "/usr/lib/python3.8/multiprocessing/connection.py", line 379, in _recv
    chunk = read(handle, remaining)
ConnectionResetError: [Errno 104] Connection reset by peer