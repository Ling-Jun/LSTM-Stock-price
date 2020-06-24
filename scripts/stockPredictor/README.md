# How to use the scripts?

* Download this folder, then navigate to the folder from CLI.
* Activate an environment with Python and the packages specified in requirements.txt.

>
 pip install -r requirements.txt

## stockPredictor.py
### train with price data
* Run

>
  python stockPredictor.py -ticker (ticker) -train y -epoch (number-of-epochs)
  -batch_size (batch_size)

* The variable '-ticker' is mandatory, '-train' has to be 'y',
  '-epochs' and '-batch_size' are optional.
* After running the command given above, there will be two files generated:

  - Model file ends with '.h5'

  - Data preparation object, e.g. scaler.pkl.

### predict price
* Run

>
  python stockPredictor.py -ticker (ticker) -train n -model (model)


* The variable '-ticker' and '-model' are mandatory, '-train' has to be 'n'.
