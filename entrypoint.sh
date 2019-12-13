#!/bin/bash
exec python create_index.py --nb_features 100000000 --output_dir index/ --index_version "$@"  

