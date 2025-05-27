ClosureInvariants
=================

ClosureInvariants is a Python package for calculating closure quantities such as closure phases and closure amplitudes using advariants and covariants derived from complex correlation data. These quantities are essential for robust interferometric analysis in radio astronomy and related fields.

Repository
----------
The source code is hosted on GitHub:  
https://github.com/nithyanandan/ClosureInvariants

Installation
------------

To install ClosureInvariants, first clone the repository:

.. code-block:: bash

    git clone https://github.com/nithyanandan/ClosureInvariants.git
    cd ClosureInvariants

Then install the package using `pip`:

.. code-block:: bash

    pip install .

Alternatively, for editable development mode:

.. code-block:: bash

    pip install -e .

Testing
-------

To run the test suite using `pytest`, ensure `pytest` and `pytest-cov` are installed. For example,

.. code-block:: bash

    pip install pytest pytest-cov

or

.. code-block:: bash

    conda install pytest pytest-cov -c conda-forge


Then run:

.. code-block:: bash

    pytest --pyargs -vv --cov ClosureInvariants

This will run all tests and generate a code coverage report.

Examples
--------

Example usage of the package is provided in the form of Jupyter notebooks 
located in the ``examples`` folder.

License
-------

This project is licensed under the MIT License.

Contributing
------------

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

Contact
-------

For questions or support, please contact `nithyanandan [dot] t [at] gmail [dot] com` or open an issue on GitHub.

