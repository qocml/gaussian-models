> regular_data.py

Generates data based on a toy model function, with noise.

The available toy model functions are a cubic function or the single qubit fidelity, specified by arguments cubic and singlequbit respectively.

The possible noise types are:
	none, specified by argument n. No further arguments are required.
	Gaussian, specified by argument g. This requires two further arguments, which are the mean and standard deviation respectively.
	binomial, specified by argument b. This requires one further argument, which is the fixed number of trials, and assumes that the success probability is decribed by the toymodel function.

To run:
	python regular_data.py <toy model type> <number of samples> <noise type> <fist noise argument if needed> <second noise argument if needed>

For example:
	python regular_data.py singlequbit 20 g 0 .05
