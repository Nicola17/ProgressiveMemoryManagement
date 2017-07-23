progressive-table
=======

Clean install
------------
Download and install Anaconda 3.6+

```bash
sudo apt-get install cmake
pip install setuptools
conda install libgcc
```

External dependencies
------------
Roaring Bitmaps
https://github.com/RoaringBitmap/CRoaring

```bash
mkdir -p build
cd build
cmake ..
make
make install
```
add in .profile

```bash
export LD_LIBRARY_PATH="/lib:/usr/lib:/usr/local/lib"
```

Install
-------------
```bash
mkdir -p build
cd build
cmake ..
make
python setup.py install
```
