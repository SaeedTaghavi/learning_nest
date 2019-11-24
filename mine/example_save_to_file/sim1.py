#!/usr/bin/env python
import nest
nest.SetKernelStatus({"overwrite_files": True})
neurons = nest.Create('iaf_neuron', 2, [{'I_e': 400.0}, {'I_e': 405.0}])
for i, n in enumerate(neurons):
    sdetector = nest.Create("spike_detector")
    nest.SetStatus(sdetector, {"withgid": True, "withtime": True, "to_file": True,"label": "send", "file_extension": "spikes"})
nest.Connect(neurons, sdetector)
nest.Simulate(500.0)
