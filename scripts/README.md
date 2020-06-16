# How to use the scripts?

* Download this folder, then navigate to the folder from CLI.
* Activate an environment with Python and the packages specified in requirements.txt.

>
 pip install -r requirements.txt

## train.py
* Run

>
  python train.py -path (data-file-path) -epoch (number-of-epcohs) -ticker (ticker-of-data-file)

* The variable '-ticker' is optional.
* After running the command given above, there will be two files generated:

  - Model file ends with '.h5'

  - Data preparation object, e.g. scaler.pkl.

## predict.py
* Run

>
  python predict.py -path (data-file-path) -model (model)
