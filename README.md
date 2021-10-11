# PartitionedFormulations_NN
This repository contains scripts for generating partition-based formulations for trained ReLU neural networks and a few test instances implemented in Gurobi. More
details on the method here: https://arxiv.org/abs/2102.04373.

Please cite this work as:
```
@article{tsay2021partition,
  title   =   {{Partition-based formulations for mixed-integer optimization of trained ReLU neural networks}},
  author  =   {Tsay, Calvin and Kronqvist, Jan and Thebelt, Alexander and Misener, Ruth},
  journal =   {ArXiv},
  volume  =   {2102.04373},
  year    =   {2021}
}
```

## Installing Gurobi
The solver software [Gurobi](https://www.gurobi.com) is required to run the examples. Gurobi is a commercial mathematical optimization solver and free of charge for academic research. It is available on Linux, Windows and Mac OS. 

Please follow the instructions to obtain a [free academic license](https://www.gurobi.com/academia/academic-program-and-licenses/). Once Gurobi is installed on your system, follow the steps to setup the Python interface [gurobipy](https://www.gurobi.com/documentation/9.0/quickstart_mac/the_grb_python_interface_f.html).

## Authors
* **[Calvin Tsay](https://www.imperial.ac.uk/people/c.tsay)** ([tsaycal](https://github.com/tsaycal)) - Imperial College London
* **[Jan Kronqvist](https://www.kth.se/profile/jankr)** ([jkronqvi](https://github.com/jkronqvi)) - KTH Royal Institute of Technology
* **[Alexander Thebelt](https://optimisation.doc.ic.ac.uk/person/alexander-thebelt/)** ([ThebTron](https://github.com/ThebTron)) - Imperial College London
* **[Ruth Misener](http://wp.doc.ic.ac.uk/rmisener/)** ([rmisener](https://github.com/rmisener)) - Imperial College London

## License
This repository is released under the Apache License 2.0. Please refer to the [LICENSE](https://github.com/cog-imperial/PartitionedFormulations_NN/blob/master/LICENSE) file for details.

## Acknowledgements
This work was supported by Engineering & Physical Sciences Research Council (EPSRC) Fellowships to CT and RM (grants EP/T001577/1 and EP/P016871/1), an Imperial College Research Fellowship to CT, a Royal So- ciety Newton International Fellowship (NIF\R1\182194) to JK, a grant by the Swedish Cultural Foundation in Finland to JK, and a PhD studentship funded by BASF to AT.

