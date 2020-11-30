for dir in data/*
do
  echo $dir
  if [[ -d $dir ]]; then
    for file in $dir/*
    do
      echo $file
      if [[ "${file: -4}" == ".sgf" ]]; then
        python preprocessing.py $file "processed/$(basename -- $file .sgf).pkl"
      fi
    done
  fi
done
