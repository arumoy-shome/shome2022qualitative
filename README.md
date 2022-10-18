# Qualitative Analysis of Data vs. Model Centric Fairness Testing

## Dependencies

The code for this project is hosted on Github. It can be downloaded as
a zip file from the [repository]. Alternatively, if you have `git`,
you can clone the repository using the following command.

	git clone --recursive --depth 1	https://github.com/arumoy-shome/shome2022qualitative.git

The datasets used for the analysis in this project can be downloaded
from [google drive]. Download the zip file & place it in the root
project directory (where this readme file is located).

The results generated from the experiments are stored in csv files
within the `data` directory. Create this directory within the root
directory of this project if it does not yet exist.

	mkdir -p data

Use the provided Dockerfile to build a docker image & run your
commands within a docker container. To build an image tagged
`shome22qual` run the following command.

	docker build -t shome22qual .

The following command mounts the current working directory to the
`/app` directory within a container & runs the
`bin/exp-feature-sets.bash` script for 5 iterations. The container is
automatically cleaned up after the script finishes executing.

	docker run --rm -it -v "$(pwd):/app" shome22qual ./bin/exp-feature-sets.bash 5


The number of cpus & memory available to the container can be
specified using the `--cpus` & `--memory` flags. The following command
limits the number of cores to 8 & the available memory to 32GB.

	docker run --rm -it -v "$(pwd):/app" --cpus 8.0 --memory 320000000000 shome22qual ./bin/exp-feature-sets.bash 5

## Organisation & Directory Structure

Following is an overview of the directory structure of this project
along with a brief description of what they contain.

    .
    ├── bin
    │   └── __pycache__
    ├── data
    ├── docs
    ├── report
    └── vendor

[aif360]: https://github.com/LINKME
[google drive]: https://LINKME
[repository]: https://github.com/arumoy-shome/shome2022qualitative/
